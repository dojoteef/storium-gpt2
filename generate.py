"""
Generate from our Storium models
"""
import sys
import logging
import argparse
from functools import partial
from typing import Any, Dict, List, Tuple
from types import SimpleNamespace
from asyncio import (
    Task,
    Queue,
    gather,
    wait_for,
    as_completed,
    ensure_future,
    new_event_loop,
    get_event_loop,
    set_event_loop,
    TimeoutError as AsyncTimeoutError,
)
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from data.dataset import StoriumDataset
from data.preprocess import SPLIT_NAMES
from data.utils import narrow
from sample import SampleGenerator
from utils import tqdm_wrap_stdout


class Scheduler:
    """
    This class does all the heavy lifting of asynchronously executing models while
    balancing the tradeoff between throughput and realtime results.
    """

    def __init__(
        self,
        generator: SampleGenerator,
        batch_size: int = 1,
        sample_length: int = 256,
        num_workers: int = 1,
        wait_time: float = 0.1,
    ):
        """
        Initialize the scheduler
        """
        self.wait_time = wait_time
        self.batch_size = batch_size
        self.sample_length = sample_length
        self.generator = generator

        self.loop = get_event_loop()
        self.queue: Queue = Queue()
        self.pool = ThreadPoolExecutor(max_workers=num_workers)
        self.workers = [ensure_future(self.main_loop()) for _ in range(num_workers)]

    async def main_loop(self):
        """ Consume a batch of tasks and execute them """
        while True:
            tasks = [await self.queue.get()]
            while len(tasks) < self.batch_size:
                try:
                    tasks.append(await wait_for(self.queue.get(), self.wait_time))
                except AsyncTimeoutError:
                    break

            futures, batch, summaries = zip(*tasks)
            results = await self.loop.run_in_executor(
                self.pool,
                partial(
                    self.generator.sample,
                    lengths=self.sample_length,
                    skip_special_tokens=False,
                ),
                summaries,
            )

            for future, result, entry, summary in zip(
                futures, results, batch, summaries
            ):
                # Set the result of the future
                future.set_result((result, entry, summary))

                # Need to notify the task queue for each item in the batch
                self.queue.task_done()

    def extract_summary(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the summaries from a list of entries by truncating the final
        segment in the entry, which is the move itself
        """
        # Use the index of the last separator to truncate the entry
        indices = (entry["tokens"] == self.generator.separator_id).nonzero().flatten()
        return narrow(entry, indices[-1] + 1)

    async def generate(
        self, entry: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """
        Schedule the figmentator to run and return the result.
        """
        future = self.loop.create_future()
        await self.queue.put((future, entry, self.extract_summary(entry)))

        return await future


class Generator:
    """
    A class that encapsulates all the functionality needed to generate from a
    model using examples from the dataset
    """

    def __init__(self, args: SimpleNamespace):
        """
        Initialize the generator
        """
        self.args = args
        self.dataset: StoriumDataset
        self.generator = SampleGenerator(
            top_k=args.sample.top_k,
            top_p=args.sample.top_p,
            temperature=args.sample.temperature,
            repetition_penalty=args.sample.repetition_penalty,
            cache_dir=args.cache_dir,
        )
        self.scheduler = Scheduler(
            self.generator,
            batch_size=args.data.batch_size,
            sample_length=args.sample.sample_length,
            num_workers=args.sample.num_workers,
        )

    def load_model(self, checkpoint_path: str):
        """
        Load the model
        """
        self.generator.load(checkpoint_path)

    def load_dataset(self, split: str):
        """
        Load the dataset
        """
        if not hasattr(self, "dataset") or self.dataset.split != split:
            logging.info("Loading %s dataset", split)
            self.dataset = StoriumDataset(split, "gpt2", cache_dir=self.args.cache_dir)
            self.dataset.load(self.args.data_dir)

    async def __call__(self):
        """
        Run the generation!
        """
        entries = self.dataset.entries
        if self.args.data.max_entries:
            entries = entries[: self.args.data.max_entries]

        batch_iterator = tqdm(
            as_completed([self.scheduler.generate(entry) for entry in entries]),
            unit="entry",
            initial=1,
            dynamic_ncols=True,
            desc="Generating",
            total=len(entries),
            file=sys.stdout,  # needed to make tqdm_wrap_stdout work
        )

        sep = "*******\n"
        with tqdm_wrap_stdout():
            example_id = 0
            for result in batch_iterator:
                sample, batch, summary = await result
                summary_length = len(summary["tokens"])
                context = self.generator.tokenizer.decode(summary["tokens"].tolist())
                original = self.generator.tokenizer.decode(
                    batch["tokens"][summary_length:].tolist()
                )
                logging.info(
                    "#%d:\n%scontext\n%s%s\n%soriginal\n%s%s\n%ssample\n%s%s",
                    example_id,
                    sep,
                    sep,
                    context,
                    sep,
                    sep,
                    original,
                    sep,
                    sep,
                    sample,
                )
                example_id += 1

            batch_iterator.close()


def define_generate_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """
    Define arguments needed for the evaluation command
    """
    parser = sub_parsers.add_parser("generate", help="Generate samples from a model")
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
        default=8,
        help="Number of examples to batch together",
    )
    data_group.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="If greater than 0, then only process up to max entries",
    )

    sample_group = parser.add_argument_group("sample")
    sample_group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="top_k > 0: keep only top k tokens with highest probability",
    )
    sample_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="top_p > 0.0: keep the top tokens with cumulative probability >= top_p"
        "Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)",
    )
    sample_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature == 0.0: greedy decoding; temperature == 1.0: normal multinomial samples",
    )
    sample_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="A repition penalty as described in CTRL (https://arxiv.org/abs/1909.05858)",
    )
    sample_group.add_argument(
        "--sample-length",
        type=int,
        default=256,
        help="The desired number of tokens to generate each sample.",
    )
    sample_group.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="How many batches to execute currently using asyncio",
    )

    parser.set_defaults(func=perform_generation)


def perform_generation(args):
    """
    Main entry point for generation
    """
    loop = new_event_loop()
    loop.set_debug(True)
    set_event_loop(loop)

    generator = Generator(args)
    generator.load_model(args.restore)
    generator.load_dataset(args.split)

    try:
        loop.run_until_complete(gather(generator()))
    finally:
        try:
            _cancel_all_tasks(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            set_event_loop(None)
            loop.close()


def _cancel_all_tasks(loop):
    to_cancel = Task.all_tasks(loop)
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(gather(*to_cancel, loop=loop, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )
