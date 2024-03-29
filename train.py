"""
Baseline training script
"""
import argparse
import copy
import glob
import json
import logging
import math
import os
import shutil
import sys
from contextlib import ExitStack
from itertools import cycle
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch

try:
    from apex import amp
except ImportError:
    pass
from comet_ml import Experiment  # must be before torch!
from torch import nn
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, AdamW, GPT2Config,
                          get_linear_schedule_with_warmup)

import metrics
from data.dataset import GPT2StoriumDataset as StoriumDataset
from data.parallel import chunked_scattering
from data.utils import get_dataloader
from evaluate import Evaluator
from experiment import initialize_experiment
from model import GPT2SegmentedModel
from utils import (collect_tensors, refresh_cuda_memory, release_cuda_memory,
                   tqdm_unwrap_stdout, tqdm_wrap_stdout)


class Trainer:
    """
    A class that encapsulates all the functionality needed to train a model
    """

    def __init__(self, args: SimpleNamespace):
        """
        Initialize the trainer
        """
        self.args = args

        self.step = 0
        self.amp_initialized = False
        self.dataset: StoriumDataset
        self.modules: Dict[str, Any] = {}
        self.experiment: Experiment

        self._initialize()
        self._initialize_metrics()

    @property
    def use_fp16(self):
        """
        Whether to use fp16 training
        """
        return "amp" in globals() and torch.cuda.is_available() and self.args.optim.fp16

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
            metrics.Metric("lr", "format_scientific", "g", max_history=1)
        )
        self.metric_store.add(
            metrics.Metric("ppl", "format_dynamic_float", max_history=1000)
        )
        self.metric_store.add(
            metrics.Metric("ntok", "format_int", "a", max_history=1000)
        )
        self.metric_store.add(metrics.Metric("oom", "format_int", "t"))
        self.metric_store.add(metrics.Metric("nll", "format_float", max_history=1000))
        self.experiment = initialize_experiment(
            self.args, ("data", "model", "optim"), self.args.experiment_name
        )

    def _initialize(self):
        """
        Load the dataset, model, etc
        """
        cache_dir = self.args.cache_dir
        model_name = self.args.model.model_name

        logging.info("Loading dataset")
        self.dataset = StoriumDataset("train", "gpt2", cache_dir=cache_dir)
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

        # Track the modules
        self.modules["model"] = model
        self.modules["optimizer"] = optimizer
        self.modules["scheduler"] = scheduler

    @property
    def checkpoint_path(self):
        """
        Return the current checkpoint path
        """
        return os.path.join(self.args.output_dir, f"checkpoint-{self.step}")

    def save(self):
        """
        Save all the tracked modules
        """
        # Save model checkpoint
        checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        logging.info("Saving model checkpoint to %s", checkpoint_path)

        train_state: Dict[str, Any] = {"step": self.step}
        if self.use_fp16:
            # Need to save the automatic mixed precision state_dict
            # See https://github.com/NVIDIA/apex#checkpointing

            # But first ensure cuda memory is relatively contiguous because the
            # call to `amp.state_dict()` seems to allocate cuda memory, which
            # can fail if cuda memory is fragmented.
            refresh_cuda_memory()
            train_state["amp"] = amp.state_dict()

        for name, module in self.modules.items():
            if name == "model":
                module.save_pretrained(checkpoint_path)
            else:
                train_state[name] = module.state_dict()

        with open(
            os.path.join(checkpoint_path, "train_state.pt"), "wb"
        ) as train_state_file:
            torch.save(train_state, train_state_file)

        with open(
            os.path.join(checkpoint_path, "train_config.json"), "wt"
        ) as train_config_file:
            json.dump(
                self.args,
                train_config_file,
                indent=2,
                default=lambda obj: getattr(obj, "__dict__", {}),
            )

        self.save_metrics()

    def save_metrics(self):
        """
        Method to save metrics to the current checkpoint path
        """
        checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        logging.info("Saving metrics to %s", checkpoint_path)
        self.metric_store.save(os.path.join(checkpoint_path, "train_metrics.json"))

    def on_new_best(self):
        """
        Mark the latest checkpoint as the best
        """
        new_best_checkpoint = os.path.join(
            self.args.output_dir, f"checkpoint-{self.step}"
        )
        logging.info("New best %s", new_best_checkpoint)
        best_checkpoint_path = os.path.join(self.args.output_dir, "best-checkpoint")
        try:
            # Remove the old best checkpoint path, otherwise it will error when
            # trying to create the symlink
            os.remove(best_checkpoint_path)
        except FileNotFoundError:
            pass

        # Just use a symlink to denote the best checkpoint
        os.symlink(
            os.path.basename(new_best_checkpoint),
            best_checkpoint_path,
        )

    def prune_checkpoints(self) -> bool:
        """
        Remove oldest checkpoints first if we are above the max checkpoints limit
        """
        if self.args.max_checkpoints <= 0:
            return False

        checkpoints = glob.glob(os.path.join(self.args.output_dir, "checkpoint-*"))
        sorted_checkpoints = sorted(
            (int(os.path.basename(c).split("-")[1]), c) for c in checkpoints
        )

        try:
            # Try to read the best checkpoint if it exists, otherwise set it to None
            best_checkpoint_path: Optional[str] = os.readlink(
                os.path.join(self.args.output_dir, "best-checkpoint")
            )
        except FileNotFoundError:
            best_checkpoint_path = None

        for _, checkpoint in sorted_checkpoints[: -self.args.max_checkpoints]:
            if os.path.basename(checkpoint) == best_checkpoint_path:
                # If the best checkpoint is about to removed, then we should
                # stop early
                logging.info("Not removing best checkpoint %s", checkpoint)
                return False

            logging.info("Removing checkpoint %s", checkpoint)
            shutil.rmtree(checkpoint)

        return True

    def load(self, checkpoint_path: str):
        """
        Load from checkpoint
        """
        train_config_path = os.path.join(checkpoint_path, "train_config.json")
        if not os.path.isfile(train_config_path):
            raise RuntimeError(f"Cannot find train config file: {train_config_path}")

        train_state_path = os.path.join(checkpoint_path, "train_state.pt")
        if not os.path.isfile(train_state_path):
            raise RuntimeError(f"Cannot find train state file: {train_state_path}")

        model_state_path = os.path.join(checkpoint_path, WEIGHTS_NAME)
        if not os.path.isfile(model_state_path):
            raise RuntimeError(f"Cannot find model state file: {model_state_path}")

        train_metrics_path = os.path.join(checkpoint_path, "train_metrics.json")
        if not os.path.isfile(train_metrics_path):
            raise RuntimeError(f"Cannot find metrics file: {train_metrics_path}")

        # Must load the train config first
        with open(train_config_path, "rt") as config_file:
            self.args = json.load(
                config_file, object_hook=lambda obj: SimpleNamespace(**obj)
            )

        train_state = torch.load(train_state_path)
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
            if self.amp_initialized:
                amp.load_state_dict(train_state["amp"])

        model_state = torch.load(model_state_path)
        for name, module in self.modules.items():
            if name == "model":
                module.load_state_dict(model_state)
            else:
                module.load_state_dict(train_state[name])

        self.step = train_state["step"]
        self.metric_store.load(train_metrics_path)

    def __call__(self):
        """
        Run the training!
        """
        # Must be called first
        self.try_init_amp()

        model = self.modules["model"]
        optimizer = self.modules["optimizer"]
        scheduler = self.modules["scheduler"]

        if self.args.optim.use_gradient_checkpointing:
            model.enable_gradient_checkpointing()

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
        accumulation_steps = self.args.optim.gradient_accumulation_steps
        progress = tqdm(
            unit="step",
            initial=self.step,
            dynamic_ncols=True,
            desc=get_description(),
            total=max_steps,
            file=sys.stdout,  # needed to make tqdm_wrap_stdout work
        )

        with ExitStack() as stack:
            # pylint:disable=no-member
            stack.enter_context(tqdm_wrap_stdout())
            stack.enter_context(chunked_scattering())
            stack.enter_context(self.experiment.train())
            # pylint:enable=no-member

            if self.args.optim.early_stopping:
                # If using early stopping, must evaluate regularly to determine
                # if training should stop early, so setup an Evaluator
                eval_args = copy.deepcopy(self.args)
                eval_args.data.batch_size = self.args.optim.eval_batch_size

                evaluator = Evaluator(eval_args)
                evaluator.model = model
                evaluator.load_dataset("validation")
                evaluator.initialize_experiment(experiment=self.experiment)

                # Make sure we are tracking validation nll
                self.metric_store.add(metrics.Metric("vnll", "format_float", "g(m)"))

                # And store a local variable for easy access
                vnll_metric = self.metric_store["vnll"]

            loss = 0
            num_tokens = 0
            for step, batch in enumerate(cycle(dataloader), 1):
                try:
                    step_loss = self.compute_gradients_and_loss(batch, model, optimizer)
                    run_optimizer = (step % accumulation_steps) == 0

                    if run_optimizer:
                        # Run an optimization step
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()

                    # Update loss and num tokens after running an optimization
                    # step, in case it results in an out of memory error
                    loss += step_loss
                    num_tokens += batch["num_tokens"]

                    if run_optimizer:
                        # Since we ran the optimizer, increment current step
                        self.step += 1
                        self.experiment.set_step(self.step)
                        progress.update()

                        # update our metrics as well
                        self.update_metrics(
                            loss / accumulation_steps,
                            num_tokens,
                            scheduler.get_lr()[0],
                        )
                        num_tokens = 0
                        loss = 0

                        # and finally check if we should save
                        if (
                            self.args.save_steps > 0
                            and self.step % self.args.save_steps == 0
                        ):
                            # First save the current checkpoint
                            self.save()

                            # Then if we are implementing early stopping, see
                            # if we achieved a new best
                            if self.args.optim.early_stopping:
                                evaluator.reset_metrics()
                                with ExitStack() as eval_stack:
                                    # pylint:disable=no-member
                                    eval_stack.enter_context(tqdm_unwrap_stdout())
                                    eval_stack.enter_context(
                                        release_cuda_memory(
                                            collect_tensors(optimizer.state)
                                        )
                                    )
                                    # pylint:enable=no-member

                                    vnll = evaluator()
                                    vnll_metric.update(vnll)

                                    # Save the updated metrics
                                    self.save_metrics()

                                    if vnll == vnll_metric.min:
                                        self.on_new_best()

                                # Try to combat OOM errors caused by doing evaluation
                                # in the same loop with training. This manifests in out
                                # of memory errors after the first or second evaluation
                                # run.
                                refresh_cuda_memory()

                            if not self.prune_checkpoints():
                                logging.info("Stopping early")
                                break

                            if self.step >= max_steps:
                                logging.info("Finished training")
                                break

                except RuntimeError as rte:
                    if "out of memory" in str(rte):
                        self.metric_store["oom"].update(1)
                        logging.warning(str(rte))
                    else:
                        progress.close()
                        raise rte

                progress.set_description_str(get_description())

            progress.close()

    def compute_gradients_and_loss(self, batch: Dict[str, Any], model, optimizer):
        """
        Compute the gradients and loss for the specified batch
        """
        model.train()
        loss = model(batch, loss_only=True)[0]

        # If there are multiple GPUs, then this will be a vector of losses, so
        # sum over the GPUs first
        loss = loss.mean()

        if self.use_fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def update_metrics(self, loss, num_tokens, lr):  # pylint:disable=invalid-name
        """
        Update the metrics
        """
        # Update our metrics
        self.metric_store["nll"].update(loss)
        self.metric_store["ntok"].update(num_tokens)
        self.metric_store["ppl"].update(math.exp(loss))
        self.metric_store["lr"].update(lr)

        # Update the experiment logs as well
        for name, metric in self.metric_store.items():
            if name == "oom":
                self.experiment.log_metric(name, metric.total)
            else:
                self.experiment.log_metric(name, metric.last_value)


def define_train_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """
    Define arguments needed for the train command
    """
    parser = sub_parsers.add_parser("train", help="Train a model")
    parser.add_argument(
        "--track",
        default=False,
        const=True,
        nargs="?",
        help="Whether to track this experiment. If an experiment id is provided, it will track \
        the existing experiment. If a filename ending with guid it is provided, it will wait \
        until the file exists, then start tracking that experiment.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="A name for the experiment when using comet for tracking",
    )
    parser.add_argument(
        "--restore",
        type=str,
        help="Restore from the specified checkpoint before continuing training",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=5000,
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
        default=2560,  # max batch size that fits on a single 2080ti using fp16
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
        default=100000,
        help="How many optimization steps to run.",
    )
    optim_group.add_argument(
        "--warmup-steps",
        type=int,
        default=8000,
        help="How many steps of warmup to apply.",
    )
    optim_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="How many steps to accumulate gradients before doing an update",
    )
    optim_group.add_argument(
        "--use-gradient-checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing. Needed for bigger models.",
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
    optim_group.add_argument(
        "--early-stopping",
        default=False,
        action="store_true",
        help="Whether to use early stopping based on validation nll",
    )
    optim_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=9 * 1024,  # Max batch size that fits on a single 2080ti
        # without going oom. This is smaller than when running evaluation
        # separately, since we need to account for additional training state
        # and fragmentation.
        help="The batch size to use for evaluation",
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
