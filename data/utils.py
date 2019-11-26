"""
Data related utilities
"""
from typing import Callable
from argparse import Namespace
from torch.utils.data import (
    Sampler,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    DataLoader,
)
from data.dataset import StoriumDataset
from data.sampler import SequenceLengthSampler


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
        batch_sampler = BatchSampler(sampler_fn(dataset), config.batch_size, False)
    else:
        raise ValueError("Unknown batch method!")

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate,
        num_workers=num_devices,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
