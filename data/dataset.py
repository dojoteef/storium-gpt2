"""
Utilities and classes for manipulating the dataset
"""
import os
import logging
import argparse
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
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

from data.preprocess import (
    Preprocessor,
    tensorize,
    SpecialToken,
    SPLIT_NAMES,
)

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

    def __init__(
        self, split: str, tokenizer_name: str, cache_dir: Optional[str] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.entries: List[Dict[str, Any]] = []

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if isinstance(idx, Sequence):
            return [self.entries[i] for i in idx]

        return self.entries[idx]

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

    def dataset_path(self, directory):
        """
        The path to the dataset
        """
        return os.path.join(directory, f"storium_{self.split}.{self.tokenizer_name}.pt")

    @staticmethod
    def _process(
        filename: str, preprocessor: Preprocessor, naive_layout: bool = False
    ) -> List[Dict[str, Any]]:
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

            with move.constraint(preprocessor.max_length, naive=naive_layout):
                entries.append(move.asdict(with_stats=True))

        return entries

    def load(self, directory):
        """
        Load the processed dataset
        """
        output_path = self.dataset_path(directory)
        if not os.path.isfile(output_path):
            raise ValueError(f"{output_path} not found!")

        self.entries = torch.load(output_path)

    def process(
        self,
        filenames: List[str],
        directory: str,
        history: int = 0,
        character_history: int = 0,
        naive_layout: bool = False,
        force: bool = False,
    ):
        """
        Process the stories from the file list and generate the tensors for the
        dataset

        - **filenames**: A list of filenames to preprocess
        - **directory**: path to a directory to save the result
        - **force**: whether to force processing if the target file already exists

        Returns the path of the preprocessed file to load
        """
        output_path = self.dataset_path(directory)
        if os.path.isfile(output_path) and not force:
            return

        if not os.path.exists(directory):
            logging.warning(
                "Output directory %s does not exist. Creating it.", directory
            )
            os.makedirs(directory)

        results = []
        pool = Pool()
        preprocessor = Preprocessor(
            self.get_tokenizer(), history=history, character_history=character_history,
        )
        for filename in filenames:
            results.append(
                pool.apply_async(
                    type(self)._process,
                    [filename, preprocessor],
                    {"naive_layout": naive_layout},
                )
                # type(self)._process(filename, preprocessor, naive_layout=naive_layout)
            )
        pool.close()

        self.entries = []
        for result in tqdm(
            results,
            unit="file",
            dynamic_ncols=True,
            desc=f"Processing {self.split} set",
        ):
            entries = result.get()
            if not entries:
                continue

            for entry in entries:
                self.entries.append(tensorize(entry))  # type: ignore
        pool.join()

        torch.save(self.entries, output_path)

    def stats_str(self):
        """
        Create a string representation of the dataset stats
        """
        count = len(self.entries)
        strings = ["dataset stats:"]
        strings.append(f" #entries={count}")

        if count:
            token_lengths = tuple(len(e["tokens"]) for e in self.entries)
            length_min = min(token_lengths)
            length_max = max(token_lengths)
            length_avg = sum(token_lengths) / count

            strings.append(
                f" length (min={length_min},avg={length_avg:.2f},max={length_max})"
            )

            segment_lengths = tuple(len(e["segments"]) for e in self.entries)
            segments_min = min(segment_lengths)
            segments_max = max(segment_lengths)
            segments_avg = sum(segment_lengths) / count

            strings.append(
                f" segments (min={segments_min},avg={segments_avg:.2f},max={segments_max})"
            )

            strings.append(" segment-level stats:")
            tokenizer = self.get_tokenizer()
            for token in SpecialToken:
                segment_id = tokenizer.convert_tokens_to_ids(token)
                segment_stats = tuple(
                    e["stats"][segment_id]
                    for e in self.entries
                    # Only count stats for the segment_id if it's actually used
                    if segment_id in e.get("stats", {})
                )
                if not segment_stats:
                    # Only include token level stats if they are actually present
                    continue

                token_min = min(segment_stats)
                token_max = max(segment_stats)
                token_avg = sum(segment_stats) / len(segment_stats)

                strings.append(
                    f"  {token} (min={token_min},avg={token_avg:.2f},max={token_max})"
                )
        return "\n".join(strings)


def perform_preprocessing(args):
    """
    Preprocess the dataset according to the passed in args
    """
    for split in SPLIT_NAMES:
        with open(os.path.join(args.data_dir, f"{split}_filenames.txt"), "rt") as file:
            filenames = [
                os.path.join(args.data_dir, filename).strip()
                for filename in file.readlines()
            ]

        dataset = StoriumDataset(split, args.tokenizer, cache_dir=args.cache_dir)
        dataset.process(
            filenames,
            args.output_dir,
            history=args.history,
            character_history=args.character_history,
            naive_layout=args.naive_layout,
            force=args.force,
        )

        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            dataset.load(args.output_dir)

        logging.info("%s %s", split, dataset.stats_str())


def define_preprocess_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """ Define the arguments needed for the preprocess command """
    parser = sub_parsers.add_parser("preprocess", help="Preprocess the dataset")
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
        "--naive-layout",
        action="store_true",
        default=False,
        help="Whether to force preprocessing if preprocessed data already exists",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Whether to force preprocessing if preprocessed data already exists",
    )
    parser.set_defaults(func=perform_preprocessing)
