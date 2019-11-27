"""
A module implementing various data samplers for datasets.
"""
import heapq
import random
from typing import Dict, List, NamedTuple, Tuple

from torch.utils.data import Sampler

from utils import ceildiv


class DeviceBucket(NamedTuple):
    """
    Metadata for a device within a particular TokenBucket
    """

    total_length: int  # must be first so heap sorts based on it
    device_id: int
    max_length: int
    sequence_ids: Tuple[int, ...] = tuple()
    sequence_lengths: Tuple[int, ...] = tuple()

    def empty(self):
        """
        Return an emptied device bucket
        """
        return DeviceBucket(0, self.device_id, self.max_length)

    def add(self, sequence_id, sequence_length):
        """
        Returns a new DeviceBucket with the given sequence added
        """
        sequence_ids = self.sequence_ids + (sequence_id,)
        sequence_lengths = self.sequence_lengths + (sequence_length,)
        total_length = max(sequence_lengths) * len(sequence_ids)

        return DeviceBucket(
            total_length,
            self.device_id,
            self.max_length,
            sequence_ids,
            sequence_lengths,
        )


class TokenBucket(object):
    """ A bucket of sequence ids """

    def __init__(self, max_lengths):
        """ Initialize the bucket """
        self.max_lengths = max_lengths
        self.reset()

    def reset(self):
        """ Reset the bucket """
        self.heap: List[DeviceBucket] = [
            DeviceBucket(0, i, length) for i, length in enumerate(self.max_lengths)
        ]

    def try_add(self, sequence_id, sequence_length):
        """ Try to add the given example """
        full = []
        while self.heap:
            device_bucket = heapq.heappop(self.heap)
            new_device_bucket = device_bucket.add(sequence_id, sequence_length)
            if new_device_bucket.total_length > device_bucket.max_length:
                full.append(device_bucket)
            else:
                heapq.heappush(self.heap, new_device_bucket)
                break

        if self.heap:
            # Add back any full device lists
            while full:
                heapq.heappush(self.heap, full.pop())
        else:
            # All batches were full
            self.reset()
            return self.extract_batch(full)

    def extract_batch(self, iterable):
        """ Extract a batch from the iterable """
        _, batch = zip(
            *sorted((bucket.device_id, bucket.sequence_ids) for bucket in iterable)
        )

        return batch

    def get_batch(self):
        """ Get the current batch """
        return self.extract_batch(self.heap)


class SequenceLengthSampler(Sampler):
    """ A sampler that tries to select batches that have a given total sequence length """

    def __init__(self, max_lengths, sequence_lengths, shuffle=False, granularity=5):
        """
        Initializer the sequence length sampler

        Inputs:
        max_lengths - a list of lengths of the desired total sequence length for each device
        lengths - a list containing the length for each example in the dataset
        """
        super(SequenceLengthSampler, self).__init__(sequence_lengths)

        self.shuffle = shuffle
        self.granularity = granularity
        self.max_lengths = max_lengths
        self.sequence_lengths = sequence_lengths

        # Initial estimate of the number of batches
        self.num_batches = ceildiv(sum(sequence_lengths), sum(max_lengths))

    def __len__(self):
        """ Estimate the number of batches per iteration """
        return self.num_batches

    def __iter__(self):
        """ Produce batches according the given lengths """
        num_batches = 0
        buckets: Dict[int, TokenBucket] = {}
        sequence_lengths = list(enumerate(self.sequence_lengths))
        if self.shuffle:
            random.shuffle(sequence_lengths)

        for idx, length in sequence_lengths:
            bucket_idx = ceildiv(length, self.granularity)
            bucket = buckets.get(bucket_idx, None)
            if not bucket:
                bucket = TokenBucket(self.max_lengths)

            batch = bucket.try_add(idx, length)
            if batch:
                # Bucket was full so yield a batch
                num_batches += 1
                yield batch

                # Add to the bucket now that it's been emptied
                bucket.try_add(idx, length)

            buckets[bucket_idx] = bucket

        # Go through all buckets to see if any can yield a batch
        for bucket in buckets.values():
            batch = bucket.get_batch()
            if all(batch):
                # Bucket had a non-empty batch left
                num_batches += 1
                yield batch

        # Update the batch estimate
        self.num_batches = num_batches
