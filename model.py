"""
Define the baseline model we'll use
"""
from typing import Any, Dict

import torch
from torch.utils import checkpoint
from transformers import GPT2LMHeadModel


def checkpointed_block(block):
    """
    Call the wrapped module
    """

    def custom_forward(x, layer_past=None, attention_mask=None, head_mask=None):
        # The checkpoint API really does not like lists, so return a tuple,
        # otherwise it errors out ¯\_(ツ)_/¯
        return tuple(block(x, layer_past, attention_mask, head_mask))

    def wrapped_block(x, layer_past=None, attention_mask=None, head_mask=None):
        return checkpoint.checkpoint(
            custom_forward, x, layer_past, attention_mask, head_mask
        )

    return wrapped_block


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

        args = {
            "inputs_embeds": self.wte(torch.max(tokens, tokens.new_zeros(1)))
            + segments_embeds.sum(1),
            "past": inputs.get("past", None),
            "labels": tokens if loss_only else None,
        }

        outputs = super().forward(**args)
        return outputs[:1] if loss_only else outputs

    def enable_gradient_checkpointing(self):
        """
        A function that enables gradient checkpointing on each of the
        transformer layers of the GPT2 model.
        """
        # pylint:disable=protected-access
        # Save off the real module list
        self.transformer._h = self.transformer.h

        # Replace the module list with wrapper functions
        del self.transformer.h  # Must first explicitly unset the variable
        self.transformer.h = [
            checkpointed_block(block) for block in self.transformer._h
        ]
        # pylint:enable=protected-access
