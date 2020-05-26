'''

    Analyze the linguistic

'''

import os, pickle
from text_analysis_utils import get_diff_score
import spacy
from spacy.tokens import Doc
import argparse
from nltk import ngrams
nlp = spacy.load('en_core_web_sm')
data = './judgement_contexts.json'


import json
import collections
from rouge import Rouge


def simple_check_doc_tokens(doc, tokens, idx, type):
  if not ( len(doc) == len(tokens) ):
    print(f'type input: {type}')
    print(f'number: {idx}')
    print(f'num spacy tokens: {len(doc)}')
    print(f'num rouge tokens: {len(tokens)}')
    print()

def custom_tokenizer(text):
    # tokens = []
    #
    # # your existing code to fill the list with tokens
    #
    # # replace this line:
    # return tokens
    tokens = text.split()
    # with this:
    return Doc(nlp.vocab, tokens)
nlp.tokenizer = custom_tokenizer

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


type_interested = 'user-kept'

def main(args):

  with open(args.data_path) as f:
    data = json.load(f)

  judgement_diffs = []

  N = args.num_grams
  ngram_counter_dict = collections.defaultdict(collections.Counter)
  for idx, item in enumerate(data):
    generated = item['generated']['description']
    finalized = item['finalized']['description']

    # copy the existing content, a dict of 4 fields
    new_dict = item
    new_dict['diffs'] = get_diff_score(generated, finalized)
    judgement_diffs.append(new_dict)

    gen_prep = rouge._preprocess_summary_as_a_whole(generated)[0]
    gen_tokens = gen_prep.split()
    gen_doc = nlp(gen_prep)
    gen_pos = [ token.pos_ for token in gen_doc]

    fin_prep = rouge._preprocess_summary_as_a_whole(finalized)[0]
    fin_tokens = fin_prep.split()
    fin_doc = nlp(fin_prep)
    fin_pos = [ token.pos_ for token in fin_doc ]

    for diff_results in new_dict['diffs']:
      if diff_results['type'] != 'user-added':
        [start, end] = diff_results['span_hypothesis']
        pos_tokens = gen_pos[start:end]
        cur_ngrams = ngrams(pos_tokens, N)
        ngram_counter_dict[diff_results['type']].update(cur_ngrams)
      elif diff_results['type'] == 'user-added':
        [start, end] = diff_results['span_reference']
        pos_tokens = fin_pos[start:end]
        cur_ngrams = ngrams(pos_tokens, N)
        ngram_counter_dict[diff_results['type']].update(cur_ngrams)
      else:
        print('Wrong type name')

    # simple_check_doc_tokens(gen_doc, gen_tokens, idx, 'generated')
    # simple_check_doc_tokens(fin_doc, fin_tokens, idx, 'finalized')
  # print(ngram_counter_dict)

  print(f'analyzed {idx + 1} user judgements')
  # print(ngram_counter.most_common(10))

  with open( os.path.join( args.output_dir, f'pos_{N}_grams.pkl'  ), 'wb') as f:
    pickle.dump(ngram_counter_dict, f)

  with open( os.path.join( args.output_dir, f'pos_{N}_grams.pkl'  ), 'rb') as f:
    pos_dict_loaded = pickle.load(f)

  print('End of main')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='./judgement_contexts.json')
  parser.add_argument('--output_dir', type=str, default='./ling_analysis')
  parser.add_argument('--num_grams', type=int, default=4)
  args = parser.parse_args()
  main(args)
  print('Done')


