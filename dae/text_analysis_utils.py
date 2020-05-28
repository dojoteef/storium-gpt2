"""
Various utilities for computing metrics
"""
import os
from typing import Dict, List, Set, Tuple
from difflib import Differ, SequenceMatcher

import aiofiles
from rouge import Rouge
import nltk
from nltk import download, word_tokenize
from nltk.metrics.agreement import AnnotationTask
nltk.download('punkt')

diff_code_mapping ={
    'table-warning': 'user-kept',
    'table-danger': 'user-removed',
    'table-success': 'user-added',
    'table-default': 'user-kept-stop'
}






differ = Differ()
stop_words: Set[str] = set()
from nltk.corpus import stopwords
stop_words.update(stopwords.words('english'))
with open(os.path.join("./ling_analysis/", "stopwords.txt"), "rt") as stopword_file:
    stop_words.update(l.strip() for l in stopword_file.readlines())



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


# async def initialize_metrics():
#     """ Initialize the metrics module """
#     # Ensure nltk has "punkt" downloaded... it's apparently needed for py-rouge
#     download("punkt")
#     await load_stopwords()
#
#
# async def load_stopwords():
#     """ Load the stopword list """
#     async with aiofiles.open(
#         os.path.join("static", "stopwords.txt"), "rt"
#     ) as stopword_file:
#         stopwords.update(l.strip() for l in await stopword_file.readlines())
#

# def remove_stopwords(text: str) -> List[str]:
#     """ Remove stop words from the given text """
#     return [token for token in word_tokenize(text) if token.lower() not in stopwords]
#


def basic_check_diff(diffs_result_list):
    for diff_dict in diffs_result_list:
        assert diff_dict['type'] in [ 'user-kept', 'user-removed', 'user-added', 'user-kept-stop']
        if diff_dict['type'] == 'user-removed':
            assert diff_dict['span_reference'] == None
        elif diff_dict['type'] == 'user-added':
            assert diff_dict['span_hypothesis'] == None
        else:
            assert (diff_dict['span_hypothesis'][1] - diff_dict['span_hypothesis'][0]) \
                   == (diff_dict['span_reference'][1] - diff_dict['span_reference'][0])
    print('basic check diff passed')



def get_diff_score(
    hypothesis: str, reference: str, gen_spacy_tokens: list,
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


    def list_find(big_list, sub_list, end_idx):
        """
        analogus method of find (which was for finding substring) for sub list
        :param big_list:
        :param sub_list:
        :return:
        """

        big_list = big_list[end_idx:]
        len_big, len_sub = len(big_list[end_idx:]), len(sub_list)
        for i in range(len_big):
            spacy_sub_str_accidental_white_space = rouge._preprocess_summary_as_a_whole(''.join(big_list[i:i + len_sub]))[0].replace('-', '')
            spacy_sub_str = spacy_sub_str_accidental_white_space.replace(' ', '')
            tokens_sub_str = ''.join(sub_list)
            # if rouge._preprocess_summary_as_a_whole(' '.join(big_list[i:i + len_sub])) == ' '.join(sub_list):
            if spacy_sub_str == tokens_sub_str:
                return i + end_idx
            else:
                return None

    def get_sublist(gen_spacy_tokens: list, start: int, tokens) -> Tuple[str, int, int, int]:
        """
        Get the original substring from the text for the given sequence of tokens
        """
        if tokens == ['where', 'jun', 'seo', 'was']:
            print('note')

        end = list_find(gen_spacy_tokens, tokens, start)
        if end == None:
            print('end = none')
        end += 1
        return ' '.join(gen_spacy_tokens[start:end]), end, start, end

    gen_spacy_tokens = [token.lower() for token in gen_spacy_tokens]

    score = 0.0
    reference_index = 0
    hypothesis_index = 0
    reference_lc = reference.lower()
    hypothesis_lc = hypothesis.lower()
    reference_tokens = rouge._preprocess_summary_as_a_whole(reference)[0].split()
    hypothesis_tokens = rouge._preprocess_summary_as_a_whole(hypothesis)[0].split()

    # diffs: List[Tuple[str, str]] = []
    matcher = SequenceMatcher(isjunk=None, a=hypothesis_tokens, b=reference_tokens)

    diffs_result_list = []
    for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
        if tag == "equal":

            diff_text, reference_index = get_substring(
                reference, reference_lc, reference_index, reference_tokens[blo:bhi]
            )

            diff_text, hypothesis_index, start, end = get_sublist(
                gen_spacy_tokens,
                hypothesis_index,
                hypothesis_tokens[alo:ahi]
            )

            diff_class = "table-default"
            if any(
                token.lower() not in stop_words for token in hypothesis_tokens[alo:ahi]
            ):
                diff_class = "table-warning"
                score += ahi - alo

            # user kept original content,
            diffs_result_list.append({
                'type': diff_code_mapping[diff_class],
                'content': hypothesis_tokens[alo:ahi],
                'span_hypothesis': [alo, ahi],
                'span_reference': [blo, bhi],
                'span_spacy': [start, end]
            })
        else:
            if tag == "delete":
                diff_text, hypothesis_index, start, end = get_sublist(
                    gen_spacy_tokens,
                    hypothesis_index,
                    hypothesis_tokens[alo:ahi]
                )
                # user deleted something, none span in reference
                diffs_result_list.append({
                    'type': diff_code_mapping[diff_class],
                    'content': hypothesis_tokens[alo:ahi],
                    'span_hypothesis': [alo, ahi],
                    'span_reference': [blo, bhi],
                    'span_spacy': [start, end]
                })
            elif tag == "insert":
                diff_text, reference_index = get_substring(
                    reference, reference_lc, reference_index, reference_tokens[blo:bhi]
                )
                # user added something, none span in hypothesis
                diffs_result_list.append({
                    'type': diff_code_mapping['table-success'],
                    'content': reference_tokens[blo:bhi],
                    'span_hypothesis': None,
                    'span_reference': [blo, bhi]
                })
            elif tag == "replace":
                diff_text, hypothesis_index, start, end = get_sublist(
                    gen_spacy_tokens,
                    hypothesis_index,
                    hypothesis_tokens[alo:ahi]
                )
                # user deleted something, none span in reference
                diffs_result_list.append({
                    'type': diff_code_mapping[diff_class],
                    'content': hypothesis_tokens[alo:ahi],
                    'span_hypothesis': [alo, ahi],
                    'span_reference': [blo, bhi],
                    'span_spacy': [start, end]
                })
                diff_text, reference_index = get_substring(
                    reference, reference_lc, reference_index, reference_tokens[blo:bhi]
                )
                # user added something, none span in hypothesis
                diffs_result_list.append({
                    'type': diff_code_mapping['table-success'],
                    'content': reference_tokens[blo:bhi],
                    'span_hypothesis': None,
                    'span_reference': [blo, bhi]
                })

        # inspection_id += 1
        # print('\n\n')

    # basic_check_diff(diffs_result_list)
    return diffs_result_list
    # pylint:enable=protected-access
