"""
Various utilities for computing metrics
"""
import os
from typing import Dict, List, Set, Tuple
from difflib import Differ, SequenceMatcher

import aiofiles
from rouge import Rouge
from nltk import download, word_tokenize
from nltk.metrics.agreement import AnnotationTask
import nltk
nltk.download('punkt')

differ = Differ()
stopwords: Set[str] = set()
rouge = Rouge(
    metrics=["rouge-n", "rouge-l", "rouge-w"],
    max_n=4,
    limit_length=False,
    apply_avg=False,
    apply_best=False,
    alpha=0.5,
    weight_factor=1.2,
    stemming=False,
)


async def initialize_metrics():
    """ Initialize the metrics module """
    # Ensure nltk has "punkt" downloaded... it's apparently needed for py-rouge
    download("punkt")
    await load_stopwords()


async def load_stopwords():
    """ Load the stopword list """
    async with aiofiles.open(
        os.path.join("static", "stopwords.txt"), "rt"
    ) as stopword_file:
        stopwords.update(l.strip() for l in await stopword_file.readlines())


def remove_stopwords(text: str) -> List[str]:
    """ Remove stop words from the given text """
    return [token for token in word_tokenize(text) if token.lower() not in stopwords]


def get_diff_score(
    hypothesis: str, reference: str
) -> Tuple[List[Tuple[str, str]], Dict[str, float]]:
    """
    Return the op code and text as a tuple.
    """

    # pylint:disable=protected-access
    def get_substring(text: str, text_lc: str, start: int, tokens) -> Tuple[str, int]:
        """
        Get the original substring from the text for the given sequence of tokens
        """
        token = ""
        end = start
        for token in tokens:
            end = text_lc.find(token, end)
        end += len(token)

        return text[start:end], end

    score = 0.0
    reference_index = 0
    hypothesis_index = 0
    reference_lc = reference.lower()
    hypothesis_lc = hypothesis.lower()
    reference_tokens = rouge._preprocess_summary_as_a_whole(reference)[0].split()
    hypothesis_tokens = rouge._preprocess_summary_as_a_whole(hypothesis)[0].split()

    diffs: List[Tuple[str, str]] = []
    matcher = SequenceMatcher(isjunk=None, a=hypothesis_tokens, b=reference_tokens)

    inspection_id = 0

    for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
        if tag == "equal":
            diff_text, hypothesis_index = get_substring(
                hypothesis, hypothesis_lc, hypothesis_index, hypothesis_tokens[alo:ahi]
            )
            diff_text, reference_index = get_substring(
                reference, reference_lc, reference_index, reference_tokens[blo:bhi]
            )
            diff_class = "table-default"
            if any(
                token.lower() not in stopwords for token in hypothesis_tokens[alo:ahi]
            ):
                diff_class = "table-warning"
                score += ahi - alo

            diffs.append((diff_class, diff_text))
        else:
            if tag == "delete":
                diff_text, hypothesis_index = get_substring(
                    hypothesis,
                    hypothesis_lc,
                    hypothesis_index,
                    hypothesis_tokens[alo:ahi],
                )
                diffs.append(("table-danger", diff_text))
            elif tag == "insert":
                diff_text, reference_index = get_substring(
                    reference, reference_lc, reference_index, reference_tokens[blo:bhi]
                )
                diffs.append(("table-success", diff_text))
            elif tag == "replace":
                diff_text, hypothesis_index = get_substring(
                    hypothesis,
                    hypothesis_lc,
                    hypothesis_index,
                    hypothesis_tokens[alo:ahi],
                )
                diffs.append(("table-danger", diff_text))
                diff_text, reference_index = get_substring(
                    reference, reference_lc, reference_index, reference_tokens[blo:bhi]
                )
                diffs.append(("table-success", diff_text))
        print( f'Inspection {inspection_id}')
        print( diffs[-1])
        print( hypothesis_index )
        print( reference_index )
        inspection_id += 1
        print('\n\n')

    return (
        diffs,
        rouge._compute_p_r_f_score(
            len(hypothesis_tokens), len(reference_tokens), score
        ),
    )
    # pylint:enable=protected-access