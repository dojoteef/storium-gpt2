"""
Various utilities
"""
import io
import sys
import contextlib
import threading
from typing import Iterable, Union
from numbers import Integral
from itertools import tee, zip_longest
from subprocess import check_output, CalledProcessError

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
