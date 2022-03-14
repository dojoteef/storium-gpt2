"""
Utilities and classes for manipulating the dataset
"""
import argparse
import logging
import os
import random

from data.preprocess import AVAILABLE_TOKENIZERS, SPLIT_NAMES

from .gpt2 import StoriumDataset as GPT2StoriumDataset
from .gpt3 import StoriumDataset as GPT3StoriumDataset


def perform_preprocessing(args):
    """
    Preprocess the dataset according to the passed in args
    """
    random.seed(args.seed)

    for split in SPLIT_NAMES:
        with open(os.path.join(args.data_dir, f"{split}_filenames.txt"), "rt") as file:
            filenames = [
                os.path.join(args.data_dir, filename).strip()
                for filename in file.readlines()
            ]

        dataset_kwargs = {"cache_dir": args.cache_dir}
        preprocessor_kwargs = {
            "force": args.force,
            "history": args.history,
            "max_length": args.max_length,
            "character_history": args.character_history,
            "preferred_entry_length": args.preferred_entry_length,
        }
        if args.dataset == "gpt2":
            dataset_cls = GPT2StoriumDataset
            preprocessor_kwargs["naive_layout"] = args.naive_layout
        elif args.dataset == "gpt3":
            dataset_cls = GPT3StoriumDataset
            if split == "train":
                # Only filter training dataset
                dataset_kwargs["max_tokens"] = args.max_tokens
                dataset_kwargs["min_completion_length"] = args.min_completion_length
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset}")

        dataset = dataset_cls(split, args.tokenizer, **dataset_kwargs)
        dataset.process(filenames, args.output_dir, **preprocessor_kwargs)

        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            dataset.load(args.output_dir)

        logging.info("%s %s", split, dataset.stats_str())


def define_preprocess_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """Define the arguments needed for the preprocess command"""
    parser = sub_parsers.add_parser("preprocess", help="Preprocess the dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("gpt2", "gpt3"),
        default="gpt2",
        help="The dataset to preprocess (gpt2 or gpt3 version)",
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
        "--max-length",
        type=int,
        default=1024,
        help="Max length for the story context + entry",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help="Max tokens in the training split (default no limit)",
    )
    parser.add_argument(
        "--min-completion-length",
        type=int,
        default=-1,
        help="Minimum entry completion length to consider for training (default no limit)",
    )
    parser.add_argument(
        "--preferred-entry-length",
        type=int,
        default=256,
        help="Desired length for the story entry",
    )
    parser.add_argument(
        "--naive-layout",
        action="store_true",
        default=False,
        help="Whether to force preprocessing if preprocessed data already exists",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Whether to force preprocessing if preprocessed data already exists",
    )
    parser.set_defaults(func=perform_preprocessing)
