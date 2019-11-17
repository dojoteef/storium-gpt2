"""
Utilities and classes for manipulating the dataset
"""
import os
import logging
import argparse
from typing import Any, Dict, List, Optional
from multiprocessing import Pool

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    TransfoXLTokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    RobertaTokenizer,
    DistilBertTokenizer,
)
import torch
from torch.utils.data import Dataset

from preprocess import (
    Preprocessor,
    tensorize,
    split_dataset,
    SpecialToken,
)


SPLIT_NAMES = ("train", "validation", "test")
AVAILABLE_TOKENIZERS = [
    model
    for tokenizer_type in (
        BertTokenizer,
        OpenAIGPTTokenizer,
        GPT2Tokenizer,
        TransfoXLTokenizer,
        XLNetTokenizer,
        XLMTokenizer,
        RobertaTokenizer,
        DistilBertTokenizer,
    )
    for model in tokenizer_type.max_model_input_sizes
]


class StoriumDataset(Dataset):
    """
    The torch dataset class for Storium for use in a DataLoader
    """

    def __init__(self, tokenizer_name: str, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.entries: List[Dict[str, Any]] = []

    @staticmethod
    def _process(filename: str, preprocessor: Preprocessor) -> List[Dict[str, Any]]:
        """
        Process a single file and return the resulting entries
        """
        entries: List[Dict[str, Any]] = []
        story = preprocessor.process_story_file(filename)
        if not story or not story.entries:
            logging.debug("Skipped %s", filename)
            return entries

        for entry_id in story.entries.keys():
            move = preprocessor.get_move(story, entry_id)
            if not move:
                continue

            with move.constraint(preprocessor.max_length):
                entries.append(move.asdict(with_stats=True))

        return entries

    def get_tokenizer(self):
        """
        Get a tokenizer for the dataset
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, cache_dir=self.cache_dir
        )

        # Cannot specify "additional_special_tokens" in the call to
        # from_pretrained, as that requires the tokens to already be in the
        # vocabulary. Rather, making a call to add_special_tokens automatically
        # adds the tokens to the vocabulary if they are not already present.
        tokenizer.add_special_tokens({"additional_special_tokens": list(SpecialToken)})

        return tokenizer

    def process_and_load(
        self,
        filenames: List[str],
        directory: str,
        split: str,
        history: int = 0,
        character_history: int = 0,
        force: bool = False,
    ):
        """
        Process the stories from the file list and generate the tensors for the
        dataset

        - **filenames**: A list of filenames to preprocess
        - **directory**: path to a directory to save the result
        - **split**: the name of the data split
        - **force**: whether to force processing if the target file already exists

        Returns the path of the preprocessed file to load
        """
        output_path = os.path.join(
            directory, f"storium_{split}.{self.tokenizer_name}.pt"
        )
        if os.path.isfile(output_path) and not force:
            self.entries = torch.load(output_path)
            return

        results = []
        pool = Pool()
        preprocessor = Preprocessor(
            self.get_tokenizer(), history=history, character_history=character_history
        )
        for filename in filenames:
            results.append(
                pool.apply_async(type(self)._process, [filename, preprocessor])
            )
        pool.close()

        self.entries = []
        for result in tqdm(
            results, unit="file", dynamic_ncols=True, desc=f"Processing {split} set"
        ):
            entries = result.get()
            if not entries:
                continue

            for entry in entries:
                self.entries.append(tensorize(entry))  # type: ignore
        pool.join()

        torch.save(self.entries, output_path)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def __repr__(self):
        count = len(self.entries)
        strings = ["dataset stats:"]
        strings.append(f" #entries={count}")

        if count:
            length_min = min(len(e["tokens"]) for e in self.entries)
            length_max = max(len(e["tokens"]) for e in self.entries)
            length_avg = sum(len(e["tokens"]) for e in self.entries) / count

            segments_min = min(len(e["segments"]) for e in self.entries)
            segments_max = max(len(e["segments"]) for e in self.entries)
            segments_avg = sum(len(e["segments"]) for e in self.entries) / count

            strings.append(
                f" length (min={length_min},avg={length_avg:.2f},max={length_max})"
            )
            strings.append(
                f" segments (min={segments_min},avg={segments_avg:.2f},max={segments_max})"
            )

            strings.append(" segment-level stats:")
            tokenizer = self.get_tokenizer()
            for token in SpecialToken:
                segment_id = tokenizer.encode(token)[0]
                token_min = min(
                    e.get("stats", {}).get(segment_id, 0) for e in self.entries
                )
                token_max = max(
                    e.get("stats", {}).get(segment_id, 0) for e in self.entries
                )
                token_avg = (
                    sum(e.get("stats", {}).get(segment_id, 0) for e in self.entries)
                    / count
                )

                if token_min or token_max or token_avg:
                    # Only include token level stats if they are actually present
                    strings.append(
                        f"  {token} (min={token_min},avg={token_avg:.2f},max={token_max})"
                    )
        return "\n".join(strings)


def perform_split(args):
    """
    Split the dataset according to the passed in args
    """
    splits = split_dataset(
        args.data_directory, (args.train_split, args.validation_split, args.test_split)
    )
    for split, filenames in zip(SPLIT_NAMES, splits):
        with open(
            os.path.join(args.output_directory, f"{split}_filenames.txt"), "wt"
        ) as split_file:
            split_file.write("\n".join(filenames))


def define_split_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """ Define the arguments needed for the split command """
    parser = sub_parsers.add_parser("split", help="Split the dataset")
    parser.add_argument(
        "--train-split",
        type=int,
        default=8,
        help="An int denoting the relative amount of data to use for training",
    )
    parser.add_argument(
        "--validation-split",
        type=int,
        default=1,
        help="An int denoting the relative amount of data to use for validation",
    )
    parser.add_argument(
        "--test-split",
        type=int,
        default=1,
        help="An int denoting the relative amount of data to use for testing",
    )
    parser.set_defaults(func=perform_split)


def perform_preprocessing(args):
    """
    Preprocess the dataset according to the passed in args
    """
    dataset = StoriumDataset(args.tokenizer, cache_dir=args.cache_dir)
    for split in SPLIT_NAMES:
        with open(
            os.path.join(args.data_directory, f"{split}_filenames.txt"), "rt"
        ) as file:
            filenames = [
                os.path.join(args.data_directory, filename).strip()
                for filename in file.readlines()
            ]

        dataset.process_and_load(
            filenames,
            args.output_directory,
            split,
            history=args.history,
            character_history=args.character_history,
            force=args.force,
        )

        logging.info("%s %s", split, dataset)


def define_preprocess_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """ Define the arguments needed for the preprocess command """
    parser = sub_parsers.add_parser("preprocess", help="Preprocess the dataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Where to cache the downloaded pretrained tokenizer",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=AVAILABLE_TOKENIZERS,
        default="gpt2",
        help="The tokenizer to use for preprocessing the dataset",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=0,
        help="How many entry summaries to include for context",
    )
    parser.add_argument(
        "--character-history",
        type=int,
        default=0,
        help="How many character specific entry summaries to include for context",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Whether to force preprocessing if preprocessed data already exists",
    )
    parser.set_defaults(func=perform_preprocessing)


def parse_args():
    """ Parse the arguments required for splitting the dataset """
    parser = argparse.ArgumentParser("Split/Preprocess Dataset")
    parser.add_argument(
        "data_directory",
        type=str,
        help="Location of the raw dataset (directory of json files)",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=".",
        help="Where to output the generated files",
    )
    sub_parsers = parser.add_subparsers()
    define_split_args(sub_parsers)
    define_preprocess_args(sub_parsers)

    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Whether to have verbose output",
    )

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_usage()
        exit(1)

    return args


def main():
    """
    Main entry point when calling this module directly
    """
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(
            logging.DEBUG if args.verbose > 1 else logging.INFO
        )

    # perform the subcommand chosen
    args.func(args)


if __name__ == "__main__":
    main()
