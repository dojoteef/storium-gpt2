"""
Our evaluation script
"""
import os
import sys
import json
import argparse
import logging
from typing import Any, Dict
from contextlib import ExitStack

import torch
from torch import nn
from tqdm import tqdm
from transformers import GPT2Config

from data.dataset import StoriumDataset
from data.utils import get_dataloader
from data.parallel import chunked_scattering
from data.preprocess import SPLIT_NAMES
from experiment import initialize_experiment
from model import GPT2SegmentedModel
from utils import tqdm_wrap_stdout
import metrics


class Evaluator:
    """
    A class that encapsulates all the functionality needed to evaluate a model
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the evaluator
        """
        self.args = args

        self.step = 0
        self.amp_initialized = False
        self.dataset: StoriumDataset
        self.model: nn.DataParallel

        self._initialize()
        self._initialize_metrics()

    def _initialize_metrics(self):
        """
        Initialize the metrics
        """
        self.metric_store = metrics.MetricStore()
        self.metric_store.add(metrics.Metric("ppl", "format_dynamic_float"))
        self.metric_store.add(metrics.Metric("ntok", "format_int", "a"))
        self.metric_store.add(metrics.Metric("nll", "format_float"))
        self.metric_store.add(metrics.Metric("oom", "format_int", "t"))

        self.experiment = initialize_experiment(self.args, ("data",))

    def _initialize(self):
        """
        Load the dataset, model, etc
        """
        cache_dir = self.args.cache_dir
        checkpoint_dir = self.args.restore

        logging.info("Loading train config")
        train_config_filename = os.path.join(checkpoint_dir, "train_config.json")
        if not os.path.isfile(train_config_filename):
            raise RuntimeError(
                f"Cannot find train config file: {train_config_filename}"
            )

        # Must load the train config first
        with open(train_config_filename, "rt") as config_file:
            train_args = json.load(
                config_file, object_hook=lambda obj: argparse.Namespace(**obj)
            )

        logging.info("Loading dataset")
        self.dataset = StoriumDataset(
            self.args.split, train_args.model.model_name, cache_dir=cache_dir
        )
        self.dataset.load(self.args.data_dir)

        logging.info("Loading model")
        config = GPT2Config.from_pretrained(self.args.restore)
        model = GPT2SegmentedModel.from_pretrained(
            self.args.restore, config=config, cache_dir=cache_dir
        )

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = nn.DataParallel(model)

    def save(self):
        """
        Save the evaluation metrics
        """
        logging.info("Saving evaluation metrics to %s", self.args.output_dir)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        self.metric_store.save(os.path.join(self.args.output_dir, "eval_metrics.json"))

    def __call__(self):
        """
        Run the evaluation!
        """
        dataloader = get_dataloader(
            self.args.data,
            self.dataset,
            num_devices=len(self.model.device_ids),
            shuffle=True,
        )

        def get_description():
            return f"Train {self.metric_store}"

        batch_iterator = tqdm(
            dataloader,
            unit="batch",
            initial=self.step + 1,
            dynamic_ncols=True,
            desc=get_description(),
            file=sys.stdout,  # needed to make tqdm_wrap_stdout work
        )

        with ExitStack() as stack:
            # pylint:disable=no-member
            stack.enter_context(tqdm_wrap_stdout())
            stack.enter_context(chunked_scattering())
            # pylint:enable=no-member

            for step, batch in enumerate(batch_iterator, self.step + 1):
                self.step = step
                self.experiment.set_step(step)

                try:
                    self.eval_step(batch)
                except RuntimeError as rte:
                    if "out of memory" in str(rte):
                        self.metric_store["oom"].update(1)
                    else:
                        batch_iterator.close()
                        raise rte

                batch_iterator.set_description_str(get_description())

            batch_iterator.close()

    def eval_step(self, batch: Dict[str, Any]):
        """
        Run a single step of evaluation using the passed in batch
        """
        self.model.eval()
        loss = self.model(batch)[0]

        # If there are multiple GPUs, then this will be a vector of losses, so
        # sum over the GPUs first
        loss = loss.mean()

        # Update our metrics
        self.metric_store["nll"].update(loss.item())
        self.metric_store["ntok"].update(batch["num_tokens"])
        self.metric_store["ppl"].update(torch.exp(loss).item())

        # Update the experiment logs as well
        for name, metric in self.metric_store.items():
            self.experiment.log_metric(name, metric.last_value)


def define_eval_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """
    Define arguments needed for the evaluation command
    """
    parser = sub_parsers.add_parser("eval", help="Train a model")
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
        "--restore",
        type=str,
        help="Restore from the specified checkpoint before evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=SPLIT_NAMES,
        help="Which dataset split to run the evaluation over",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=2560,  # max batch size that fits on a single 2080ti using fp16
        help="The batch size to use for evaluation",
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

    parser.set_defaults(func=perform_eval)


def perform_eval(args):
    """
    Main entry point for eval
    """
    evaluator = Evaluator(args)
    evaluator()

    evaluator.save()
