"""
Our evaluation script
"""
import os
import sys
import argparse
import logging
from typing import Any, Dict, Optional
from types import SimpleNamespace
from contextlib import ExitStack

from comet_ml import Experiment  # must be before torch!
import torch
from tqdm import tqdm
from transformers import GPT2Config

from data.dataset import StoriumDataset
from data.utils import get_dataloader
from data.parallel import chunked_scattering, StaticDataParallel
from data.preprocess import SPLIT_NAMES
from experiment import initialize_experiment
from model import GPT2SegmentedModel
from utils import tqdm_wrap_stdout
import metrics


class Evaluator:
    """
    A class that encapsulates all the functionality needed to evaluate a model
    """

    def __init__(self, args: SimpleNamespace):
        """
        Initialize the evaluator
        """
        self.args = args

        self.best_nll = float("inf")
        self.amp_initialized = False
        self.dataset: StoriumDataset
        self.model: StaticDataParallel
        self.experiment: Experiment

        self.reset_metrics()

    def reset_metrics(self):
        """
        Initialize the metrics
        """
        self.metric_store = metrics.MetricStore()
        self.metric_store.add(metrics.Metric("ppl", "format_dynamic_float"))
        self.metric_store.add(metrics.Metric("ntok", "format_int", "a"))
        self.metric_store.add(metrics.Metric("nll", "format_float"))
        self.metric_store.add(metrics.Metric("oom", "format_int", "t"))

    def initialize_experiment(self, experiment: Optional[Experiment] = None):
        """
        Initialize the experiment
        """
        self.experiment = (
            initialize_experiment(self.args, ("data",), self.args.experiment_name)
            if experiment is None
            else experiment
        )

    def load(self, checkpoint_dir):
        """
        Load the model, etc
        """
        logging.info("Loading model")
        config = GPT2Config.from_pretrained(checkpoint_dir)
        model = GPT2SegmentedModel.from_pretrained(
            checkpoint_dir, config=config, cache_dir=self.args.cache_dir
        )

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = StaticDataParallel(model)

    def load_dataset(self, split: str):
        """
        Load the dataset
        """
        if not hasattr(self, "dataset") or self.dataset.split != split:
            logging.info("Loading %s dataset", split)
            self.dataset = StoriumDataset(split, "gpt2", cache_dir=self.args.cache_dir,)
            self.dataset.load(self.args.data_dir)

    def save(self):
        """
        Save the evaluation metrics
        """
        logging.info("Saving evaluation metrics to %s", self.args.output_dir)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        self.metric_store.save(os.path.join(self.args.output_dir, "eval_metrics.json"))

    def __call__(self) -> float:
        """
        Run the evaluation!
        """
        dataloader = get_dataloader(
            self.args.data, self.dataset, num_devices=len(self.model.device_ids)
        )

        def get_description():
            return f"Eval {self.metric_store}"

        batch_iterator = tqdm(
            dataloader,
            unit="batch",
            initial=1,
            dynamic_ncols=True,
            desc=get_description(),
            file=sys.stdout,  # needed to make tqdm_wrap_stdout work
        )

        with ExitStack() as stack:
            # pylint:disable=no-member
            stack.enter_context(tqdm_wrap_stdout())
            stack.enter_context(chunked_scattering())
            # pylint:enable=no-member

            for batch in batch_iterator:
                try:
                    self.eval_step(batch)
                except RuntimeError as rte:
                    if "out of memory" in str(rte):
                        self.metric_store["oom"].update(1)
                        logging.warning(str(rte))
                    else:
                        batch_iterator.close()
                        raise rte

                batch_iterator.set_description_str(get_description())

            batch_iterator.close()

        return self.metric_store["nll"].average

    def eval_step(self, batch: Dict[str, Any]):
        """
        Run a single step of evaluation using the passed in batch
        """
        with torch.no_grad():
            self.model.eval()
            loss = self.model(batch, loss_only=True)[0]

        # If there are multiple GPUs, then this will be a vector of losses, so
        # sum over the GPUs first
        loss = loss.mean()

        # Update our metrics
        self.metric_store["nll"].update(loss.item())
        self.metric_store["ntok"].update(batch["num_tokens"])
        self.metric_store["ppl"].update(torch.exp(loss).item())

    def log_experiment(self):
        """
        Log the experiment metrics
        """
        if self.dataset.split == "train":
            # Do not update experiment logs if running evaluation over the
            # training set
            return

        experiment_mode = (
            self.experiment.validate
            if self.dataset.split == "validation"
            else self.experiment.test
        )
        with experiment_mode():
            for name, metric in self.metric_store.items():
                self.experiment.log_metric(name, metric.average)


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
        "--experiment-name",
        type=str,
        help="A name for the experiment when using comet for tracking",
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
        default=11 * 1024,  # max batch size that fits on a single 2080ti
        # without going oom.
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
    evaluator.load(args.restore)
    evaluator.load_dataset(args.split)
    evaluator.initialize_experiment()
    evaluator()

    evaluator.save()
    evaluator.log_experiment()
