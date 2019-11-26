"""
Baseline training script

This is the training script for our baseline model. We will generate at most
256 tokens of an entry at test time, so we allow the context to be at most 768
tokens, as GPT2 supports a maximum length of 1024 tokens for backpropagation.

Our baseline model supports using the following information as context:
- Cards played
- Character information
- Challenge being addressed
- Location card of the current scene
"""
import os
import sys
import glob
import json
import shutil
import argparse
import logging
from typing import Any, Dict
from itertools import cycle, islice
from contextlib import ExitStack

import torch
from apex import amp
from torch import nn
from tqdm import tqdm
from transformers import (
    AdamW,
    GPT2Config,
    WEIGHTS_NAME,
    get_linear_schedule_with_warmup,
)

from data.dataset import StoriumDataset
from data.utils import get_dataloader
from data.parallel import chunked_scattering
from model import GPT2SegmentedModel
from utils import tqdm_wrap_stdout
import metrics


class Trainer:
    """
    A class that encapsulates all the functionality needed to train a model
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the trainer
        """
        self.args = args

        self.step = 0
        self.amp_initialized = False
        self.dataset: StoriumDataset
        self.modules: Dict[str, Any] = {}

        self._initialize()
        self._initialize_metrics()

    @property
    def use_fp16(self):
        """
        Whether to use fp16 training
        """
        return torch.cuda.is_available() and self.args.optim.fp16

    def try_init_amp(self):
        """
        Due to the way NVIDIA's apex library works you can only call initialize
        once. This leads to a chicken-and-egg problem, when trying to restore
        a checkpoint to continue training.

        That's why we track whether or not we called initialize, such that it
        is safe to call this method multiple times (which can happen if we load
        from a checkpoint that used automatic mixed precision training).
        """
        if not self.amp_initialized and self.use_fp16:
            model = self.modules["model"]
            optimizer = self.modules["optimizer"]
            model, optimizer = amp.initialize(
                model.cuda(), optimizer, opt_level=self.args.optim.fp16_opt_level
            )
            self.modules["model"] = model
            self.modules["optimizer"] = optimizer
            self.amp_initialized = True

    def _initialize_metrics(self):
        """
        Initialize the metrics
        """
        self.metric_store = metrics.MetricStore()
        self.metric_store.add(
            metrics.Metric("ppl", metrics.format_dynamic_float, max_history=1000)
        )
        self.metric_store.add(
            metrics.Metric("loss", metrics.format_float, max_history=1000)
        )
        self.metric_store.add(
            metrics.Metric("ntok", metrics.format_int, "a", max_history=1000)
        )

    def _initialize(self):
        """
        Load the dataset, model, etc
        """
        cache_dir = self.args.cache_dir
        model_name = self.args.model.model_name

        logging.info("Loading dataset")
        self.dataset = StoriumDataset("train", model_name, cache_dir=cache_dir)
        self.dataset.load(self.args.data_dir)

        # By default the config outputs "past", but that makes our chunked
        # scattering (needed when batching based on tokens, rather than
        # examples) fail since the huggingface/transformers package stacks the
        # outputs on dim 0, which is normally the batch dimension. This leads
        # to errors like:
        #
        # RuntimeError: Gather got an input of invalid size: got [2, 5, 12,
        #   411, 64], but expected [2, 4, 12, 411, 64] (gather at
        #   /pytorch/torch/csrc/cuda/comm.cpp:226)
        #
        # During training we only care about the loss, so just disable all
        # additional outputs.
        config = GPT2Config.from_pretrained(model_name, cache_dir=cache_dir)
        config.output_hidden_states = False
        config.output_attentions = False
        config.output_past = False

        model = GPT2SegmentedModel.from_pretrained(
            model_name, config=config, cache_dir=cache_dir
        )

        tokenizer = self.dataset.get_tokenizer()
        model.resize_token_embeddings(len(tokenizer))

        max_steps = self.args.optim.max_steps
        optimizer = AdamW(model.parameters(), lr=self.args.optim.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=max_steps,
            num_warmup_steps=self.args.optim.warmup_steps,
        )

        if self.use_fp16:
            model, optimizer = amp.initialize(
                model.cuda(), optimizer, opt_level=self.args.optim.fp16_opt_level
            )

        # Track the modules
        self.modules["model"] = model
        self.modules["optimizer"] = optimizer
        self.modules["scheduler"] = scheduler

    def save(self):
        """
        Save all the tracked modules
        """
        # Save model checkpoint
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.step}",)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        logging.info("Saving model checkpoint to %s", checkpoint_dir)

        train_state: Dict[str, Any] = {"step": self.step}
        if self.use_fp16:
            # Need to save the automatic mixed precision state_dict
            # See https://github.com/NVIDIA/apex#checkpointing
            train_state["amp"] = amp.state_dict()

        for name, module in self.modules.items():
            if name == "model":
                module.save_pretrained(checkpoint_dir)
            else:
                train_state[name] = module.state_dict()

        with open(
            os.path.join(checkpoint_dir, "train_state.pt"), "wb"
        ) as train_state_file:
            torch.save(train_state, train_state_file)

        with open(
            os.path.join(checkpoint_dir, "train_config.json"), "wt"
        ) as train_config_file:
            json.dump(
                self.args,
                train_config_file,
                indent=2,
                default=lambda obj: getattr(obj, "__dict__", {}),
            )

        self.prune_checkpoints()

    def prune_checkpoints(self):
        """
        Remove oldest checkpoints first if we are above the max checkpoints limit
        """
        if self.args.max_checkpoints <= 0:
            return

        checkpoints = glob.glob(os.path.join(self.args.output_dir, "checkpoint-*"))
        sorted_checkpoints = sorted(
            (int(os.path.basename(c).split("-")[1]), c) for c in checkpoints
        )

        for _, checkpoint in sorted_checkpoints[: -self.args.max_checkpoints]:
            logging.info("Removing checkpoint %s", checkpoint)
            shutil.rmtree(checkpoint)

    def load(self, checkpoint_dir: str):
        """
        Load from checkpoint
        """
        train_config_filename = os.path.join(checkpoint_dir, "train_config.json")
        if not os.path.isfile(train_config_filename):
            raise RuntimeError(f"Cannot find config file: {train_config_filename}")

        train_state_filename = os.path.join(checkpoint_dir, "train_state.pt")
        if not os.path.isfile(train_state_filename):
            raise RuntimeError(f"Cannot find train state file: {train_state_filename}")

        model_state_filename = os.path.join(checkpoint_dir, WEIGHTS_NAME)
        if not os.path.isfile(model_state_filename):
            raise RuntimeError(f"Cannot find model state file: {model_state_filename}")

        # First load the train config
        with open(train_config_filename, "rt") as config_file:
            self.args = json.load(
                config_file, object_hook=lambda obj: argparse.Namespace(**obj)
            )

        train_state = torch.load(train_state_filename)
        if "amp" in train_state:
            # Need to load the automatic mixed precision state_dict. Calling
            # amp.load_state_dict requires initializing automatic mixed
            # precision first.
            #
            # See https://github.com/NVIDIA/apex#checkpointing
            self.try_init_amp()

            # Also, for some reason, amp.load_state_dict needs to be before
            # loading the rest of the state dicts, otherwise amp keeps the
            # params on the cpu. Not sure why this happens, as the
            # documentation seems to indicate you should call
            # amp.load_state_dict last...
            amp.load_state_dict(train_state["amp"])

        model_state = torch.load(model_state_filename)
        for name, module in self.modules.items():
            if name == "model":
                module.load_state_dict(model_state)
            else:
                module.load_state_dict(train_state[name])

        self.step = train_state["step"]

    def __call__(self):
        """
        Run the training!
        """
        # Must be called first
        self.try_init_amp()

        model = self.modules["model"]
        optimizer = self.modules["optimizer"]
        scheduler = self.modules["scheduler"]

        model = nn.DataParallel(model)
        dataloader = get_dataloader(
            self.args.data,
            self.dataset,
            num_devices=len(model.device_ids),
            shuffle=True,
        )

        def get_description():
            return f"Train {self.metric_store}"

        max_steps = self.args.optim.max_steps
        batches = islice(cycle(dataloader), max_steps - self.step)
        batch_iterator = tqdm(
            batches,
            unit="batch",
            initial=self.step + 1,
            dynamic_ncols=True,
            desc=get_description(),
            total=max_steps,
            file=sys.stdout,  # needed to make tqdm_wrap_stdout work
        )

        with ExitStack() as stack:
            # pylint:disable=no-member
            stack.enter_context(tqdm_wrap_stdout())
            stack.enter_context(chunked_scattering())
            # pylint:enable=no-member

            for step, batch in enumerate(batch_iterator, self.step + 1):
                self.step = step
                self.train_step(batch, model, optimizer, scheduler)
                batch_iterator.set_description_str(get_description())

                if self.args.save_steps > 0 and step % self.args.save_steps == 0:
                    self.save()

    def train_step(self, batch: Dict[str, Any], model, optimizer, scheduler):
        """
        Run a single step of training using the passed in batch
        """
        model.train()
        loss = model(batch)[0]

        # If there are multiple GPUs, then this will be a vector of losses, so
        # sum over the GPUs first
        loss = loss.mean()

        if self.args.optim.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()

        # Update our metrics
        self.metric_store["loss"].update(loss.item())
        self.metric_store["ntok"].update(batch["num_tokens"])
        self.metric_store["ppl"].update(torch.exp(loss).item())


def define_train_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """
    Define arguments needed for the train command
    """
    parser = sub_parsers.add_parser("train", help="Train a model")
    parser.add_argument(
        "--restore",
        type=str,
        help="Restore from the specified checkpoint before continuing training",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save after every n number of steps",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="The max number of checkpoints to keep",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        choices=GPT2SegmentedModel.pretrained_model_archive_map.keys(),
        help="The location of the processed data",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="The batch size to use for training",
    )
    data_group.add_argument(
        "--batch-size-buffer",
        type=int,
        default=0,
        help="By how many tokens to reduce the batch size on the GPU of the optimizer",
    )
    data_group.add_argument(
        "--batch-method",
        type=str,
        default="token",
        choices=["token", "example"],
        help="Whether to batch by individual examples or by number of tokens",
    )
    data_group.add_argument(
        "--token-bucket-granularity",
        type=int,
        default=3,
        help="Granularity of each bucket for the token based batching method",
    )

    optim_group = parser.add_argument_group("optim")
    optim_group.add_argument(
        "--learning-rate",
        dest="lr",
        type=float,
        default=5e-5,
        help="The initial learning rate",
    )
    optim_group.add_argument(
        "--max-steps",
        type=int,
        default=500000,
        help="How many optimization steps to run.",
    )
    optim_group.add_argument(
        "--warmup-steps",
        type=int,
        default=8000,
        help="How many steps of warmup to apply.",
    )
    optim_group.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Whether to use 16-bit floats if available using NVIDIA apex.",
    )
    optim_group.add_argument(
        "--fp16-opt-level",
        type=str,
        default="O1",
        choices=[f"O{i}" for i in range(4)],
        help="What optimization level to use for fp16 floats. "
        "See https://nvidia.github.io/apex/amp.html#opt-levels",
    )

    parser.set_defaults(func=perform_training)


def perform_training(args):
    """
    Main entry point for training
    """
    trainer = Trainer(args)
    if args.restore:
        trainer.load(args.restore)

    trainer()
