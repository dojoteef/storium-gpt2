"""
Define the baseline model we'll use
"""
import inspect
from typing import Any, Dict

import torch
from torch import nn
from torch.utils import checkpoint
from transformers import GPT2LMHeadModel


class CheckpointedModule(nn.Module):
    """
    Wrapper around an nn.Module which implements gradient checkpointing
    """

    def __init__(self, module: nn.Module):
        super().__init__()

        self.as_list = False
        self.module = module
        self.params = tuple(inspect.signature(module.forward).parameters.values())

    def __len__(self):
        """
        Ask wrapped module for len
        """
        return len(self.module)  # type: ignore

    def __iter__(self):
        """
        Ask wrapped module for an iterator
        """
        return iter(self.module)  # type: ignore

    def __getattr__(self, name):
        """
        If this method gets called, it means an attribute was not found on this
        wrapper object, so we should look to the wrapped module to find that attribute.
        """
        module = super().__getattr__("module")
        if name == "module":
            return module

        return getattr(module, name)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """ Simply call the wrapped module's state_dict method """
        return self.module.state_dict(destination, prefix, keep_vars)

    def get_args(self, args, kwargs):
        """ Fill in defaults """
        all_args = {}
        for idx, param in enumerate(self.params):
            # Set the argument to its default value first (if it has one)
            if param.default != param.empty:
                all_args[param.name] = param.default

            # Override default value with any specified args
            if idx < len(args):
                all_args[param.name] = args[idx]

            # Finally, override any specified keyword args
            if param.name in kwargs:
                all_args[param.name] = kwargs[param.name]

        return tuple(all_args.values())

    def forward(self, *args, **kwargs):  # pylint:disable=arguments-differ
        retval = checkpoint.checkpoint(
            self.checkpointed_forward, *self.get_args(args, kwargs)
        )
        if self.as_list:
            # If the huggingface/transformers code expects a list, convert the
            # output from the function call from a tuple back to a list.
            self.as_list = False  # reset for the next call to forward
            return list(retval)

        return retval

    def checkpointed_forward(self, *args):
        """ Run the module """
        retval = self.module(*args)
        if isinstance(retval, list):
            # Some modules return a list, but the checkpoint API really does
            # not like lists, so return a tuple instead, otherwise it errors
            # out ¯\_(ツ)_/¯. Apparently, the underlying torch._C._FunctionBase
            # that checkpointing is built on expects the return value to be a
            # tuple of tensors...
            self.as_list = True
            return tuple(retval)

        return retval


class GPT2SegmentedModel(GPT2LMHeadModel):
    """
    Our baseline model which uses composable segments
    """

    @property
    def wte(self):
        """
        Get the weights for the token embeddings from the transformer
        """
        return self.transformer.wte

    def forward(
        self, inputs: Dict[str, Any], loss_only=False
    ):  # pylint:disable=arguments-differ
        """
        Compose the segments together and call the base class. Also include an
        argument to control whether to only output the loss. By default the
        huggingface/transformer models output their hidden states as well,
        which is a lot of data to transfer, and thus slows down
        training/evaluation.
        """
        tokens = inputs["tokens"]
        segment_masks = inputs["segment_masks"].unsqueeze(-1)
        segments_embeds = self.wte(inputs["segments"]) * segment_masks

        kwargs = {
            "inputs_embeds": self.wte(torch.max(tokens, tokens.new_zeros(1)))
            + segments_embeds.sum(1),
            "past": inputs.get("past", None),
            "labels": tokens if loss_only else None,
        }

        outputs = super().forward(**kwargs)
        return outputs[:1] if loss_only else outputs

    def enable_gradient_checkpointing(self, level=1):
        """
        A function that enables gradient checkpointing for the GPT2 model.
        """
        if level == 1:
            for idx in range(len(self.transformer.h)):
                self.transformer.h[idx] = CheckpointedModule(self.transformer.h[idx])

        if level >= 2:
            # Needed for training GPT-2 large on 2080Ti GPUs
            module_stack = [self]

            # Store off the transformer module, because we wrap it in a
            # CheckpointedModule below
            transformer = self.transformer
            while module_stack:
                parent_module = module_stack.pop()
                for name, module in parent_module.named_children():
                    if parent_module == transformer and (
                        name == "wpe" or name == "wte"
                    ):
                        # These modules provide embeddings for the inputs, and
                        # seem to require normal gradients for the call to
                        # backward() on the loss to work
                        continue

                    setattr(parent_module, name, CheckpointedModule(module))
                    module_stack.append(module)

