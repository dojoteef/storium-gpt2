"""
Utilities and classes for manipulating the dataset
"""
import json
import logging
import os
import sys
from contextlib import ExitStack
from typing import Any, AnyStr, Dict, List, Optional, Pattern, Sequence

from torch.utils.data import Dataset
from tqdm import tqdm

from data.preprocess import get_tokenizer
from data.preprocess.gpt3 import Preprocessor


class StoriumDataset(Dataset):
    """
    The torch dataset class for Storium for use in a DataLoader
    """

    def __init__(
        self,
        model_filter: Pattern[AnyStr],
        tokenizer_name: str,
        cache_dir: Optional[str] = None,
        max_tokens: int = -1,
        min_completion_length: int = -1,
    ):
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.entries: List[Dict[str, Any]] = []

        self.model_filter = model_filter
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
        return os.path.join(directory, "storium_train.gpt3edits.jsonl")

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
        filename: str,
        directory: str,
        force: bool = False,
        max_edit_length: int = 2048,
        **preprocessor_kwargs,
    ):
        """
        Process the stories from the file list and generate the tensors for the
        dataset

        - **filename**: The filename with the edits to preprocess
        - **directory**: path to a directory to save the result
        - **force**: whether to force processing if the target file already exists
        - **max_edit_length**: max length for an edit

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

        preprocessor = Preprocessor(self.get_tokenizer(), **preprocessor_kwargs)
        with open(filename, "rt") as file:
            edits = json.load(file)

        total_length = 0
        for edit in tqdm(
            edits,
            unit="edit",
            dynamic_ncols=True,
            desc="Processing edits",
            file=sys.stderr,
        ):
            model_name = edit.get("model_name")
            if not model_name or not self.model_filter.match(model_name):
                continue

            raw_story = edit.get("story")
            generated = edit.get("generated")
            finalized = edit.get("finalized")
            if not raw_story or not generated or not finalized:
                continue

            story = preprocessor.process_story(raw_story)
            generated_entry_info, finalized_entry_info = preprocessor.process_edit(
                raw_story, generated, finalized
            )
            context, edit = preprocessor.get_edit(
                story, generated_entry_info, finalized_entry_info
            )
            if not edit:
                continue

            with ExitStack() as stack:
                stack.enter_context(context.constraint(preprocessor.max_length))
                stack.enter_context(edit.constraint(max_edit_length))

                prompt = preprocessor.decode(context)
                completion = preprocessor.decode(edit)

                # Determine the real length of the sequence, so choose an arbitrarily large
                # max_length such that the sequence isn't truncated
                length = len(
                    preprocessor.encode(prompt + completion, max_length=999999)
                )
                prompt_length = len(preprocessor.encode(prompt, max_length=999999))
                completion_length = len(
                    preprocessor.encode(completion, max_length=999999)
                )

                if (
                    self.min_completion_length > 0
                    and completion_length < self.min_completion_length
                ):
                    continue

                self.entries.append(
                    {
                        "length": length,
                        "story": filename,
                        "prompt": prompt,
                        "completion": completion,
                        "prompt_length": prompt_length,
                        "completion_length": completion_length,
                    }
                )
                total_length += length
                if total_length >= self.max_tokens:
                    break

        if not self.entries:
            logging.warning("%s has no edits", filename)

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
