"""
Various utilities
"""
import gc
import io
import sys
import contextlib
import threading
from typing import Dict, Iterable, List, Mapping, Sequence, Union
from numbers import Integral
from itertools import chain, tee, zip_longest
from subprocess import check_output, CalledProcessError

import torch
from torch import nn
from tqdm import tqdm


def ceildiv(x: Union[Integral, int], y: Union[Integral, int]):
    """ See https://stackoverflow.com/a/17511341 """
    return -(-x // y)


def pairwise(iterable: Iterable, longest: bool = False):
    """
    See itertools recipes:

    https://docs.python.org/3/library/itertools.html#itertools-recipes
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    x, y = tee(iterable)
    next(y, None)
    if longest:
        return zip_longest(x, y)

    return zip(x, y)


def grouper(iterable, n, fillvalue=None, padded=False):  # pylint:disable=invalid-name
    """
    See itertools recipes:
    https://docs.python.org/3/library/itertools.html#itertools-recipes
    Collect data into fixed-length chunks or blocks
    """
    args = [iter(iterable)] * n
    groups = zip_longest(*args, fillvalue=fillvalue)
    if padded:
        # keep the fill value
        return groups
    else:
        # ignore the fill value
        return [[x for x in group if x is not fillvalue] for group in groups]


class TQDMStreamWrapper(io.IOBase):
    """ A wrapper around an existing IO stream to funnel to tqdm """

    def __init__(self, stream):
        """ Initialize the stream wrapper """
        super(TQDMStreamWrapper, self).__init__()
        self.stream = stream

    def write(self, line):
        """ Potentially write to the stream """
        if line.rstrip():  # avoid printing empty lines (only whitespace)
            tqdm.write(line, file=self.stream)


_STREAMS = threading.local()
_STREAMS.stdout_stack = []


@contextlib.contextmanager
def tqdm_wrap_stdout():
    """ Wrap a sys.stdout and funnel it to tqdm.write """
    _STREAMS.stdout_stack.append(sys.stdout)
    sys.stdout = TQDMStreamWrapper(sys.stdout)  # type:ignore
    yield
    sys.stdout = _STREAMS.stdout_stack.pop()


@contextlib.contextmanager
def tqdm_unwrap_stdout():
    """ Unwrap a tqdm.write and funnel it to sys.stdout """
    saved = sys.stdout
    sys.stdout = _STREAMS.stdout_stack.pop()
    yield
    _STREAMS.stdout_stack.append(sys.stdout)
    sys.stdout = saved


def get_version_string():
    """ Return a git version string for the repo """
    try:
        version = check_output(
            ["git", "describe", "--always", "--dirty"], encoding="utf-8"
        )
    except CalledProcessError:
        raise RuntimeError('Call to "git describe" failed!')

    return version


@contextlib.contextmanager
def release_cuda_memory(tensors: List[torch.Tensor]):
    """
    A context manager that moves the memory for the entire module from GPU to
    CPU for the duration of the operation.
    """
    locations: Dict[torch.Tensor, torch.device] = {}
    for tensor in tensors:
        locations[tensor] = tensor.device
        tensor.data = tensor.data.cpu()
        if isinstance(tensor, nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.cpu()

    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()

    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)


def collect_tensors(collection: Union[torch.Tensor, Sequence, Mapping]):
    """
    Collect all the tensors in the sequence/mapping
    """
    if isinstance(collection, torch.Tensor):
        return [collection]

    if isinstance(collection, Sequence):
        return list(chain.from_iterable(collect_tensors(c) for c in collection))

    if isinstance(collection, Mapping):
        return list(
            chain.from_iterable(collect_tensors(v) for v in collection.values())
        )

    return []


@contextlib.contextmanager
def release_cuda_memory(tensors: List[torch.Tensor]):
    """
    A context manager that moves the memory for the entire module from GPU to
    CPU for the duration of the operation.
    """
    locations: Dict[torch.Tensor, torch.device] = {}
    for tensor in tensors:
        locations[tensor] = tensor.device
        tensor.data = tensor.data.cpu()
        if isinstance(tensor, nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.cpu()

    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()

    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)


def refresh_cuda_memory():
    """
    Essentially resets all cuda memory to help with fragmentation related
    issues.

    Fragmentation appears to be worsened by including both evaluation and
    training together in the same loop.
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then refresh the memory while also clearing the cuda cache
    with release_cuda_memory(
        [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
    ):
        pass
