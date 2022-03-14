"""
Utilities and classes for manipulating the dataset
"""
import logging
import os
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.preprocess import get_tokenizer, tensorize
from data.preprocess.gpt2 import Preprocessor, SpecialToken


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
        return get_tokenizer(self.tokenizer_name, cache_dir=self.cache_dir)

    def dataset_path(self, directory):
        """
        The path to the dataset
        """
        return os.path.join(directory, f"storium_{self.split}.{self.tokenizer_name}.pt")

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

        for entry_info in story.entries.values():
            move = preprocessor.get_move(story, entry_info)
            if not move:
                continue

            with move.constraint(
                preprocessor.max_length, naive=preprocessor.naive_layout
            ):
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
        force: bool = False,
        **preprocessor_kwargs,
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
        preprocessor = Preprocessor(self.get_tokenizer(), **preprocessor_kwargs)
        for filename in filenames:
            results.append(
                pool.apply_async(type(self)._process, [filename, preprocessor])
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
