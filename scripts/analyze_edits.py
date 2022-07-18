#!/usr/bin/env python

"""
Various utilities for computing metrics
"""
import argparse
import dataclasses
import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizer


@dataclasses.dataclass
class EditMetrics:
    """Metrics for Storium edits"""

    first_equal_idx: Optional[int] = None
    last_equal_idx: Optional[int] = None


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser("Analyze Storium Edits")
    parser.add_argument("edits_path", type=str, help="Path to the edits file")
    parser.add_argument(
        "-s", "--stopwords-path", type=str, help="Path to stopwords list"
    )
    parser.add_argument("-m", "--model-filter", type=str, help="Regex for model filter")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Where to cache the downloaded pretrained config/tokenizer/model",
    )

    return parser.parse_args()


def load_stopwords(stopwords_path) -> Set[str]:
    """Load the stopword list"""
    if not os.path.exists(stopwords_path):
        return set()

    with open(stopwords_path, "rt") as stopword_file:
        return set(w.strip() for w in stopword_file.readlines())


def get_edit_metrics(
    tokenizer: PreTrainedTokenizer,
    generated: str,
    finalized: str,
    stopwords: Set[str] = set(),
) -> EditMetrics:
    """
    Return the op code and text as a tuple.
    """
    generated_tokens = re.findall(r"\w+|[^\w\s]+", generated)
    finalized_tokens = re.findall(r"\w+|[^\w\s]+", finalized)

    # Get around extraneous logging issue
    logger = logging.getLogger(PreTrainedTokenizer.__module__)
    log_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    # Use a bogus max length to prevent the text from being truncated
    token_ids = tokenizer.encode(finalized, max_length=999999)
    logger.setLevel(log_level)

    def advance_token_indices(
        token: str, token_idx: int, text_idx: int
    ) -> Tuple[int, int]:
        # Advance the indices to reflect the more recent token
        def find_idx():
            # Have to set clean_up_tokenization_spaces to 'False' to get around this bug:
            # https://github.com/huggingface/transformers/issues/3846
            return tokenizer.decode(
                token_ids[:token_idx], clean_up_tokenization_spaces=False
            )[text_idx:].find(token)

        while token_idx < len(token_ids) and find_idx() < 0:
            token_idx += 1

        return token_idx, text_idx + find_idx() + len(token)

    text_idx = 0
    token_idx = 0
    edit_metrics = EditMetrics()
    matcher = SequenceMatcher(isjunk=None, a=generated_tokens, b=finalized_tokens)
    for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
        if tag == "equal":
            indices: List[int] = []
            for i in range(blo, bhi):
                token = finalized_tokens[i]
                token_idx, text_idx = advance_token_indices(token, token_idx, text_idx)
                if token.lower() not in stopwords and re.match(r"\w+", token):
                    indices.append(token_idx)

            if indices:
                edit_metrics.last_equal_idx = indices[-1]
                if edit_metrics.first_equal_idx is None:
                    edit_metrics.first_equal_idx = indices[0]
        elif tag == "insert" or tag == "replace":
            for i in range(blo, bhi):
                token_idx, text_idx = advance_token_indices(
                    finalized_tokens[i], token_idx, text_idx
                )

    return edit_metrics


def compute_stats(edit_metrics: List[EditMetrics], field: str) -> Dict[str, Any]:
    """Computes min, max, avg for the given field"""
    metrics_dicts = [dataclasses.asdict(e) for e in edit_metrics]

    def metrics_iter():
        """An iterator over the field metrics"""
        for e in metrics_dicts:
            if e[field] is not None:
                yield e[field]

    count = sum(1 for e in metrics_iter())
    mean = sum(metrics_iter()) / count
    var = sum((m - mean) ** 2 for m in metrics_iter()) / count
    return {
        f"{field} (count)": count,
        f"{field} (min)": min(metrics_iter()),
        f"{field} (max)": max(metrics_iter()),
        f"{field} (mean)": mean,
        f"{field} (std)": var ** 0.5,
    }


def main():
    """The main script entry point for analyzing the edits"""
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
    model_filter = re.compile(args.model_filter) if args.model_filter else None
    stopwords = load_stopwords(args.stopwords_path) if args.stopwords_path else set()
    with open(args.edits_path, "rt") as edits_file:
        edit_metrics: List[EditMetrics] = [
            get_edit_metrics(
                tokenizer,
                edit["generated"]["description"],
                edit["finalized"]["description"],
                stopwords=stopwords,
            )
            for edit in json.load(edits_file)
            if model_filter and model_filter.match(edit["model_name"])
        ]

    print(f"***EDIT STATS (n={len(edit_metrics)})***")
    for field in dataclasses.fields(EditMetrics):
        print(
            "\n".join(
                [f"{k}={v}" for k, v in compute_stats(edit_metrics, field.name).items()]
            )
        )


if __name__ == "__main__":
    main()
