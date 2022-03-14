"""
This file contains a number of utility methods for preprocessing stories.
"""
import argparse
import bisect
import glob
import heapq
import json
import logging
import os
from enum import Enum, auto
from numbers import Number
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, TypeVar, Union)

import torch
from transformers import AutoTokenizer, GPT2Tokenizer

SPLIT_NAMES = ("train", "validation", "test")
AVAILABLE_TOKENIZERS = [
    model
    for tokenizer_type in (GPT2Tokenizer,)
    for model in tokenizer_type.max_model_input_sizes
]
DataType = TypeVar("DataType")


class Trim(Enum):
    """
    An enum denoting how to trim a Segment that is too long

    - **start**: trim the start of the segment
    - **end**: trim the end of the segment
    - **middle**: trim the middle of the segment
    - **none**: do not trim the segment
    """

    start = auto()
    end = auto()
    middle = auto()


class IndexedSet(List[DataType]):
    """
    A class that makes indexing a unique sorted list easy. All the entries must
    have unique keys, if you try to insert an already existing key, it will
    raise an error.

    Loosely based on SortedCollection, which is referenced in the python docs
    for bisect.

    See: https://code.activestate.com/recipes/577197-sortedcollection/
    """

    def __init__(self, *iterable: Iterable[DataType], key=int):
        super().__init__()
        self._key = key
        self._keys: List[Any] = []

        # Ensure the list is in sorted order by inserting one at a time
        for value in tuple(*iterable):
            self.insert(value)

    def insert(self, value):
        """
        Insert into the set
        """
        key = self._key(value)
        idx = bisect.bisect_left(self._keys, key)
        if (
            idx != len(self._keys)
            and self[idx] == value  # pylint:disable=unsubscriptable-object
        ):
            # it's already in the set, no need to insert it
            return

        self._keys.insert(idx, key)
        super().insert(idx, value)  # pylint:disable=no-member

    def index(self, value: DataType) -> int:  # type: ignore
        """
        Find the index of the item in the set
        """
        key = self._key(value)
        idx = bisect.bisect_left(self._keys, key)
        if (
            idx != len(self._keys)
            and self[idx] == value  # pylint:disable=unsubscriptable-object
        ):
            return idx

        raise ValueError(f"{value} not in set")


class IndexedDict(Dict[str, DataType]):
    """
    A convenient wrapper around dict that allows integer-based indexing
    operations
    """

    indices: List[str]
    reverse_indices: Dict[str, int]

    def __init__(
        self, mapping: Union[Iterable[Tuple[str, DataType]], Mapping[str, DataType]]
    ):
        super().__init__()
        self.indices = []
        self.reverse_indices = {}

        if isinstance(mapping, Mapping):
            mapping = mapping.items()

        for idx, (key, value) in enumerate(mapping):
            self.indices.append(key)
            self.reverse_indices[key] = idx
            super().__setitem__(key, value)  # pylint:disable=no-member

    def __reduce__(self):
        return (type(self), (dict(self),))

    def __delitem__(self, key: str):
        raise RuntimeError("IndexedDict is immutable!")

    def __setitem__(self, key: str, value: DataType):
        raise RuntimeError("IndexedDict is immutable!")

    def index(self, key: str) -> int:
        """
        Return the index of the key
        """
        try:
            return self.reverse_indices[key]
        except KeyError:
            raise ValueError(f"{key} not in dict")


def tensorize(
    nested: Union[Sequence, Mapping]
) -> Union[Sequence, Mapping, torch.Tensor]:
    """
    Convert the potentially nested sequence or mapping of ints to torch tensors
    """
    if not nested:
        return nested

    if isinstance(nested, Sequence):
        element = nested[0]
        if isinstance(element, Number):
            return torch.tensor(nested)  # pylint:disable=not-callable
        elif isinstance(element, Sequence) or isinstance(element, Mapping):
            return type(nested)(tensorize(e) for e in nested)  # type: ignore
    elif isinstance(nested, Mapping):
        element = next(iter(nested.values()))
        if isinstance(element, Sequence) or isinstance(element, Mapping):
            return {k: tensorize(v) for k, v in nested.items()}

    return nested


def get_tokenizer(tokenizer_name: str, cache_dir: Optional[str] = None):
    """
    Get a tokenizer for the dataset
    """
    return AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)


def split_dataset(data_path, splits: Tuple[int, ...]) -> Tuple[List[str], ...]:
    """
    Return a list of files that split the dataset according to these constraints:

    1) They approximately meet the passed in splits, which are treated as
    ratios by taking splits[i]/sum(splits)
    2) We balance the number of stories and token counts for each split
    according to the ratios

    NOTE: The number of words in an entry is based upon splitting along
    whitespace boundaries. This is to divorce the dataset splits from any
    particular tokenization scheme, e.g. GPT2
    """
    story_info = []
    total_files = 0.0
    total_words = 0.0
    total_scenes = 0.0
    total_entries = 0.0
    data_path = os.path.abspath(data_path)
    for filename in glob.glob(os.path.join(data_path, "**/*.json"), recursive=True):
        with open(filename, "rt") as file:
            story = json.load(file)

        num_words = 0
        num_entries = 0
        scenes = story.get("scenes", [])
        for scene in scenes:
            entries = scene.get("entries", [])
            num_entries += len(entries)
            for entry in entries:
                description = entry.get("description", "")

                # We do a very simple tokenization that simply splits on whitespace to
                # get a ballpark estimate of the length of the story, only looking at
                # written entries (not cards, challenges, etc since they make up a
                # small fraction of the total written text in a story). This
                # works well enough in practice to get decently balanced
                # splits.
                num_words += len(description.split())

        num_scenes = len(scenes)

        total_files += 1
        total_words += num_words
        total_scenes += num_scenes
        total_entries += num_entries

        story_info.append(
            (
                num_words,
                num_entries,
                num_scenes,
                filename,
            )
        )

    class Split:
        """
        A class that encapsulates a split and allows for comparisons
        """

        def __init__(self, idx: int, ratio: float):
            self.words = 0
            self.scenes = 0
            self.entries = 0

            self.idx = idx
            self.ratio = ratio
            self.filenames: List[str] = []

        def add(self, words: int, entries: int, scenes: int, filename: str):
            """Add a file to the split"""
            self.words += words
            self.scenes += scenes
            self.entries += entries
            self.filenames.append(filename.replace(data_path, "").lstrip("/"))

        @property
        def weight(self) -> float:
            """Return the 'weight' of the split"""
            return self.words / self.ratio

        def __lt__(self, other: Any) -> bool:
            return (
                self.weight < other.weight  # type:ignore
                if isinstance(other, Split)
                else NotImplemented
            )

        def __str__(self) -> str:
            num_files = len(self.filenames)
            return f"Split #{self.idx}: " + ", ".join(
                [
                    f"words={self.words} ({self.words/total_words:.2f})",
                    f"entries={self.entries} ({self.entries/total_entries:.2f})",
                    f"scenes={self.scenes} ({self.scenes/total_scenes:.2f})",
                    f"files={num_files} ({num_files/total_files:.2f})",
                ]
            )

    # Create the priority queue for splits. The heap invariant is the "weight"
    # of the split, but at the start no split has any files, so the order of
    # the heap is arbitrary. Sort based on ratio to make for a deterministic
    # splitting given the same ratios.
    divisor = float(sum(splits))
    split_queue: List[Split] = sorted(
        [Split(idx, split / divisor) for idx, split in enumerate(splits)],
        key=lambda s: s.ratio,
        reverse=True,
    )

    # Do a reverse sort based on the words, entries, and scenes such that we
    # handle the largest stories first. This should give better
    # packing/balancing of splits.
    for words, entries, scenes, filename in sorted(story_info, reverse=True):
        best_split = heapq.heappop(split_queue)
        best_split.add(words, entries, scenes, filename)
        heapq.heappush(split_queue, best_split)

    # Put the splits back into the original order specified for the input
    final_splits = sorted(split_queue, key=lambda s: s.idx)
    for split in final_splits:
        logging.info(split)

    return tuple(s.filenames for s in final_splits)


def perform_split(args):
    """
    Split the dataset according to the passed in args
    """
    splits = split_dataset(
        args.data_dir, (args.train_split, args.validation_split, args.test_split)
    )
    for split, filenames in zip(SPLIT_NAMES, splits):
        with open(
            os.path.join(args.output_dir, f"{split}_filenames.txt"), "wt"
        ) as split_file:
            # Technically POSIX requires text files to end in a newline...
            split_file.write("\n".join(filenames) + "\n")


def define_split_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """Define the arguments needed for the split command"""
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
