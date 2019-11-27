"""
A module to handle metrics
"""
import copy
import json
import numbers
from functools import partial
from typing import Any, Dict

import numpy as np

from utils import pairwise


def format_time(seconds):
    """ Format time in h:mm:ss.ss format """
    hour = 60 * 60
    hours = int(seconds // hour)
    minutes = int((seconds % hour) // 60)
    return f"{hours}:{minutes:>02}:{seconds % 60.:>05.2f}"


def format_basic(value, format_spec=""):
    """ Wrapper around format() for use in functools.partial """
    return format(value, format_spec)


def format_dynamic(value, format_funcs=(format_basic,)):
    """
    Wrapper around a number of format functions that chooses the shortest
    output string.
    """
    return sorted((f(value) for f in format_funcs), key=len)[0]


# pylint:disable=invalid-name
format_int = partial(format_basic, format_spec=".0f")
format_percent = partial(format_basic, format_spec=".1%")
format_float = partial(format_basic, format_spec=".1f")
format_scientific = partial(format_basic, format_spec=".3g")
format_dynamic_float = partial(
    format_dynamic, format_funcs=(format_float, format_scientific)
)
# pylint:enable=invalid-name


FORMATTERS = {
    "format_int": format_int,
    "format_time": format_time,
    "format_basic": format_basic,
    "format_dynamic": format_dynamic,
    "format_percent": format_percent,
    "format_float": format_float,
    "format_scientific": format_scientific,
    "format_dynamic_float": format_dynamic_float,
}


class Metric:
    """ Class that represents a metric """

    def __init__(
        self,
        name,
        formatter="format_basic",
        default_format_str="g(a)",
        max_history=None,
    ):
        self.name = name
        self.max_history = max_history

        self._formatter = formatter
        self.default_format_str = default_format_str

        self.counts, self.values, self.min, self.max = self.reset()

    @classmethod
    def from_dict(cls, state: Dict[str, Any]):
        """
        Create a metric from the passed in dictionary
        """
        metric = Metric("")
        metric.__dict__.update(state)

        return metric

    @property
    def formatter(self):
        """
        Get the formatter function for this metric
        """
        return FORMATTERS[self._formatter]

    def reset(self):
        """ Reset the metrics """
        self.counts = []
        self.values = []
        self.min = float("inf")
        self.max = float("-inf")
        return self.counts, self.values, self.min, self.max

    def update(self, value, count=1):
        """ Update the value and counts """
        self.counts.append(count)
        self.values.append(value)

        average = value / count
        self.min = min(self.min, average)
        self.max = max(self.max, average)

        if self.max_history and len(self.counts) > self.max_history:
            self.counts = self.counts[1:]
            self.values = self.values[1:]

    def updates(self, values, counts=1):
        """ Update multiple values at once """
        if isinstance(counts, numbers.Number):
            counts = [counts] * len(values)

        self.counts.extend(counts)
        self.values.extend(values)
        if self.max_history:
            # pylint thinks self.max_history is None...
            # pylint:disable=invalid-unary-operand-type
            self.counts = self.counts[-self.max_history :]
            self.values = self.values[-self.max_history :]
            # pylint:enable=invalid-unary-operand-type

        averages = [value / count for count, value in zip(counts, values)]
        self.min = min(self.min, min(averages))
        self.max = max(self.max, max(averages))

    @property
    def last_count(self):
        """ Return the last recorded count of the metric"""
        # fancy way to return the last count or zero
        return len(self.counts) and self.counts[-1]

    @property
    def last_value(self):
        """ Return the last recorded value of the metric """
        # fancy way to return the last value or zero
        return len(self.values) and self.values[-1]

    @property
    def last_average(self):
        """ Return the last recorded value of the metric """
        # fancy way to return the last value or zero
        return self.last_value / max(self.last_count, 1)

    @property
    def total(self):
        """ Return the current total """
        return sum(self.values)

    @property
    def total_count(self):
        """ Return the current total count """
        return sum(self.counts)

    @property
    def average(self):
        """ Return the current average value """
        return self.total / max(self.total_count, 1)

    @property
    def var(self):
        """ Return the variance of the values """
        # Need to use a weighted average since each value has an associated count
        counts = np.array(self.counts)
        values = np.array(self.values)
        weights = counts / self.total_count
        return np.average((values - self.average) ** 2, weights=weights)

    @property
    def std(self):
        """ Return the standard deviation of the values """
        return np.sqrt(self.var)

    def __format__(self, format_str):
        """ Return a formatted version of the metric """
        format_str = format_str or self.default_format_str
        formatted = f"{self.name}="

        compact = True
        paren_depth = 0
        for format_spec, next_format_spec in pairwise(format_str, True):
            if format_spec == "l":
                compact = False
            elif format_spec == "c":
                compact = True
            elif format_spec == "(":
                formatted += "("
                paren_depth += 1
            elif format_spec == ")":
                formatted += ")"
                paren_depth -= 1
            elif format_spec == "C":
                if not compact:
                    formatted += f"last_count="
                formatted += f"{self.formatter(self.last_count)}"
            elif format_spec == "V":
                if not compact:
                    formatted += f"last_value="
                formatted += f"{self.formatter(self.last_value)}"
            elif format_spec == "g":
                if not compact:
                    formatted += f"last_avg="
                formatted += f"{self.formatter(self.last_average)}"
            elif format_spec == "a":
                if not compact:
                    formatted += f"avg="
                formatted += f"{self.formatter(self.average)}"
            elif format_spec == "t":
                if not compact:
                    formatted += f"total="
                formatted += f"{self.formatter(self.total)}"
            elif format_spec == "m":
                if not compact:
                    formatted += f"min="
                formatted += f"{self.formatter(self.min)}"
            elif format_spec == "x":
                if not compact:
                    formatted += f"max="
                formatted += f"{self.formatter(self.max)}"
            elif format_spec == "s":
                if not compact:
                    formatted += f"std="
                formatted += f"{self.formatter(self.std)}"
            elif format_spec == "v":
                if not compact:
                    formatted += f"var="
                formatted += f"{self.formatter(self.var)}"
            else:
                raise ValueError(f"Unknown format specifier {format_spec}")

            if paren_depth and format_spec != "(" and next_format_spec != ")":
                formatted += ","
                if not compact:
                    formatted += " "

        return formatted

    def __str__(self):
        """ Return a string representation of the metric """
        return self.__format__(self.default_format_str)


class MetricStore(object):
    """ A collection of metrics """

    def __init__(self, default_format_str="c"):
        super(MetricStore, self).__init__()

        self.metrics = {}
        self.default_format_str = default_format_str

    def keys(self):
        """ Return the metrics keys """
        return self.metrics.keys()

    def values(self):
        """ Return the metrics values """
        return self.metrics.values()

    def items(self):
        """ Return the metrics items """
        return self.metrics.items()

    def __getitem__(self, key):
        """ Return the requested metric """
        return self.metrics[key]

    def __contains__(self, key):
        """ See if we are tracking the named metric """
        return key in self.metrics

    def __len__(self):
        """ Count of the metrics being tracked """
        return len(self.metrics)

    def add(self, metrics):
        """ Adds a copy of the Metrics to the store """
        if isinstance(metrics, Metric):
            # if you pass a single metric
            self.metrics[metrics.name] = metrics
        else:
            # metrics must otherwise be iterable
            self.metrics.update(
                {metric.name: copy.deepcopy(metric) for metric in metrics}
            )

    def save(self, path):
        """ Save the metrics to disk """
        with open(path, "wt") as metric_file:
            json.dump(
                self.metrics,
                metric_file,
                indent=2,
                default=lambda obj: getattr(obj, "__dict__", {}),
            )

    def load(self, path):
        """ Load the metrics from disk """
        with open(path, "rt") as metric_file:
            for name, metric_state in json.load(metric_file).items():
                self.metrics[name] = Metric.from_dict(metric_state)

    def __str__(self):
        """ Return a string representation of the metric store """
        return self.__format__(self.default_format_str)

    def __format__(self, format_str):
        """ Return a formatted version of the metric """
        format_str = format_str or self.default_format_str

        if format_str == "l":
            return "\n".join(str(m) for m in self.metrics.values())
        else:
            return ", ".join(str(m) for m in self.metrics.values())
