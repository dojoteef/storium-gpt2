"""
Data related utilities
"""
from typing import Any, Callable, Dict, Sequence, Union
from argparse import Namespace
from itertools import chain

import torch
from torch import nn
from torch.utils.data import (
    Sampler,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    DataLoader,
)
from data.dataset import StoriumDataset
from data.sampler import SequenceLengthSampler


EntryList = Sequence[Dict[str, Any]]


def collate(entries: Union[Sequence[EntryList], EntryList]) -> Dict[str, Any]:
    """
    Collate the list of batches
    """

    def collator(entry_list: EntryList):
        """
        Collate a list of entries
        """
        tokens = nn.utils.rnn.pad_sequence(
            tuple(e["tokens"] for e in entry_list), batch_first=True, padding_value=-1,
        )
        max_length = tokens.shape[-1]

        def pad(entry, field):
            """
            Pad the segment
            """
            return nn.utils.rnn.pad_sequence(
                tuple(
                    torch.cat(
                        (s[field], s[field].new_full((max_length - len(s[field]),), 0),)
                    )
                    for s in entry["segments"]
                ),
                batch_first=True,
                padding_value=0,
            )

        segments = nn.utils.rnn.pad_sequence(
            tuple(pad(e, "segments") for e in entry_list),
            batch_first=True,
            padding_value=0,
        )
        segment_masks = nn.utils.rnn.pad_sequence(
            tuple(pad(e, "mask") for e in entry_list),
            batch_first=True,
            padding_value=0,
        )

        lengths = [len(e["tokens"]) for e in entry_list]
        return {
            "tokens": tokens,
            "segments": segments,
            "segment_masks": segment_masks,
            "lengths": lengths,
            "num_tokens": sum(lengths),
        }

    if not entries:
        return {}

    if isinstance(entries[0], Sequence):
        collated = collator(list(chain(*entries)))
        collated["chunk_sizes"] = tuple(len(e) for e in entries)
    else:
        collated = collator(entries)  # type:ignore

    return collated


def narrow(entry: Dict[str, Any], length: int, dim: int = 0) -> Dict[str, Any]:
    """
    Shorten the given entry to the specified length
    """

    def narrow_map(obj):
        if isinstance(obj, torch.Tensor):
            return obj.narrow(dim, 0, length)
        if isinstance(obj, tuple) and obj:
            return tuple(map(narrow_map, obj))
        if isinstance(obj, list) and obj:
            return list(map(narrow_map, obj))
        if isinstance(obj, dict) and obj:
            return type(obj)(map(narrow_map, obj.items()))
        return obj

    # After narrow_map is called, a narrow_map cell will exist. This cell
    # has a reference to the actual function narrow_map, which has references
    # to a closure that has a reference to the narrow_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return narrow_map(entry)
    finally:
        narrow_map = None  # type:ignore


def get_dataloader(
    config: Namespace,
    dataset: StoriumDataset,
    worker_init_fn: Callable[[int], None] = lambda x: None,
    pin_memory: bool = True,
    num_devices: int = 1,
    shuffle: bool = False,
):
    """ Utility function that gets a data loader """
    batch_sampler: Sampler
    if config.batch_method == "token":
        # Calculate batch sizes for each device. Potentially reduce the batch size on device 0 as
        # the optimization step (all the gradients from all devices) happens on device 0.
        batch_sizes = [config.batch_size - config.batch_size_buffer]
        batch_sizes += [config.batch_size] * (num_devices - 1)
        batch_sampler = SequenceLengthSampler(
            batch_sizes,
            [len(e["tokens"]) for e in dataset.entries],
            shuffle=shuffle,
            granularity=config.token_bucket_granularity,
        )
    elif config.batch_method == "example":
        sampler_fn = RandomSampler if shuffle else SequentialSampler
        batch_sampler = BatchSampler(
            sampler_fn(dataset), num_devices * config.batch_size, False
        )
    else:
        raise ValueError("Unknown batch method!")

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate,
        num_workers=1,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
