"""
Generate from our Storium models
"""
import sys
import logging
import argparse
from typing import Any, Dict, List
from types import SimpleNamespace

from tqdm import tqdm

from data.dataset import StoriumDataset
from data.preprocess import SPLIT_NAMES
from data.utils import EntryList, narrow
from sample import SampleGenerator
from utils import grouper, tqdm_wrap_stdout


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

    def extract_summary(self, entries: EntryList) -> List[Dict[str, Any]]:
        """
        Extract the summaries from a list of entries by truncating the final
        segment in the entry, which is the move itself
        """
        entry_list: List[Dict[str, Any]] = []
        for entry in entries:
            # Use the index of the last separator to truncate the entry
            indices = (
                (entry["tokens"] == self.generator.separator_id).nonzero().flatten()
            )
            entry_list.append(narrow(entry, indices[-1] + 1))

        return entry_list

    def __call__(self):
        """
        Run the generation!
        """
        entries = self.dataset.entries
        if self.args.data.max_entries:
            entries = entries[: self.args.data.max_entries]

        batch_iterator = tqdm(
            grouper(entries, self.args.data.batch_size),
            unit="batch",
            initial=1,
            dynamic_ncols=True,
            desc="Generating",
            file=sys.stdout,  # needed to make tqdm_wrap_stdout work
        )

        sep = "*******\n"
        with tqdm_wrap_stdout():
            for batch_idx, batch in enumerate(batch_iterator):
                summaries = self.extract_summary(batch)
                samples = self.generator.sample(summaries)
                for idx, sample in enumerate(samples):
                    summary_length = len(summaries[idx]["tokens"])
                    context = self.generator.tokenizer.decode(
                        summaries[idx]["tokens"].tolist()
                    )
                    original = self.generator.tokenizer.decode(
                        batch[idx]["tokens"][summary_length:].tolist()
                    )
                    logging.info(
                        "#%d:\n%scontext\n%s%s\n%soriginal\n%s%s\n%ssample\n%s%s",
                        batch_idx + idx,
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

    parser.set_defaults(func=perform_generation)


def perform_generation(args):
    """
    Main entry point for generation
    """
    generator = Generator(args)
    generator.load_model(args.restore)
    generator.load_dataset(args.split)
    generator()
