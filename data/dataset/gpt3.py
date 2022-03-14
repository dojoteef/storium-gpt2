"""
Utilities and classes for manipulating the dataset
"""
import json
import logging
import os
import sys
from itertools import chain
from multiprocessing import Pool
from random import randrange
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import Dataset
from tqdm import tqdm

from data.preprocess import get_tokenizer
from data.preprocess.gpt3 import Preprocessor, SpecialToken


class StoriumDataset(Dataset):
    """
    The torch dataset class for Storium for use in a DataLoader
    """

    def __init__(
        self,
        split: str,
        tokenizer_name: str,
        cache_dir: Optional[str] = None,
        max_tokens: int = -1,
        min_completion_length: int = -1,
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.entries: List[Dict[str, Any]] = []

        self.max_tokens = max_tokens
        self.min_completion_length = min_completion_length

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
        return os.path.join(directory, f"storium_{self.split}.gpt3.jsonl")

    @staticmethod
    def _process(filename: str, preprocessor: Preprocessor) -> List[Dict[str, Any]]:
        """
        Process a single file and return the resulting entries
        """
        entries: List[Dict[str, Any]] = []
        story = preprocessor.process_story_file(filename)
        if not story or not story.entries:
            logging.warning("Skipped empty story file - %s", filename)
            return entries

        for entry_info in story.entries.values():
            move = preprocessor.get_move(story, entry_info)
            if not move:
                continue

            with move.constraint(preprocessor.max_length):
                entry = preprocessor.decode(move)

                # Then determine where to split the sequence into prompt/completion
                split_point = entry.rfind(SpecialToken.current_move) + len(
                    SpecialToken.current_move
                )

                prompt = entry[:split_point]
                completion = entry[split_point:]

                # Determine the real length of the sequence, so choose an arbitrarily large
                # max_length such that the sequence isn't truncated
                length = len(preprocessor.encode(entry, max_length=999999))
                prompt_length = len(preprocessor.encode(prompt, max_length=999999))
                completion_length = len(
                    preprocessor.encode(completion, max_length=999999)
                )

                entries.append(
                    {
                        "length": length,
                        "story": filename,
                        "prompt": prompt,
                        "completion": completion,
                        "prompt_length": prompt_length,
                        "completion_length": completion_length,
                    }
                )

        return entries

    def load(self, directory):
        """
        Load the processed dataset
        """
        output_path = self.dataset_path(directory)
        if not os.path.isfile(output_path):
            raise ValueError(f"{output_path} not found!")

        with open(output_path, "rt") as file:
            self.entries = [json.loads(line) for line in file]

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

        all_stories = []
        for result in tqdm(
            results,
            unit="file",
            dynamic_ncols=True,
            desc=f"Processing {self.split} set",
            file=sys.stderr,
        ):
            all_stories.append(result.get())
        pool.join()

        if self.max_tokens <= 0:
            self.entries = list(chain.from_iterable(all_stories))
        else:
            # Increase diversity by selecting one entry from each story before getting a
            # second entry
            total_length = 0
            while all_stories and total_length < self.max_tokens:
                story_entries = all_stories.pop(0)
                if story_entries:
                    entry = story_entries.pop(randrange(len(story_entries)))
                    if (
                        self.min_completion_length <= 0
                        or entry["completion_length"] >= self.min_completion_length
                    ):
                        total_length += entry["length"]
                        self.entries.append(entry)

                if story_entries:
                    all_stories.append(story_entries)

        with open(output_path, "wt") as file:
            for entry in self.entries:
                json.dump(entry, file)
                file.write("\n")

    def stats_str(self):
        """
        Create a string representation of the dataset stats
        """
        count = len(self.entries)
        strings = ["dataset stats:"]
        strings.append(f" #entries={count}")

        if count:
            token_lengths = tuple(e["length"] for e in self.entries)
            length_min = min(token_lengths)
            length_max = max(token_lengths)
            length_avg = sum(token_lengths) / count

            strings.append(
                f" length (min={length_min},avg={length_avg:.2f},max={length_max})"
            )

        return "\n".join(strings)
