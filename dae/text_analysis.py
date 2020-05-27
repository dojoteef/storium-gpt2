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
  user_kept_pos_ngram_counter = collections.Counter()
  all_gen_pos_ngram_counter = collections.Counter()
  pos_textlist_dict = collections.defaultdict(list)

  gen_len = 0
  gen_ent_len = 0
  kept_len = 0
  kept_ent_len = 0
  num_user_kept = 0

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

    ents = gen_doc.ents
    gen_ent_ids = []
    for ent in ents:
      ent_ids = list(range(ent.start, ent.end))
      gen_ent_ids.extend(ent_ids)
    gen_len += len(gen_pos)
    gen_ent_len += len(gen_ent_ids)


    # fin_prep = rouge._preprocess_summary_as_a_whole(finalized)[0]
    # fin_tokens = fin_prep.split()
    # fin_doc = nlp(fin_prep)
    # fin_pos = [ token.pos_ for token in fin_doc ]

    all_gen_pos_ngram_counter.update( ngrams(gen_pos, N) )  # update the count of POS ngrams

    for diff_results in new_dict['diffs']:
      if diff_results['type'] == type_interested:
        [start, end] = diff_results['span_hypothesis']
        content = diff_results['content']

        cur_ngrams_kept = list( ngrams(content, N) ) # a list of n grams of actual text
        pos_tokens = gen_pos[start:end]
        cur_ngrams_pos = list( ngrams(pos_tokens, N) ) # a list of n grams of pos
        user_kept_pos_ngram_counter.update(cur_ngrams_pos) # update the count of POS ngrams

        assert len(content) == len(pos_tokens)
        assert len(cur_ngrams_kept) == len(cur_ngrams_pos)

        # update the dictionary (str, list): (pos_ngram, list of actual ngrams as examples)
        for pos_ngram, text_ngram in zip(cur_ngrams_pos, cur_ngrams_kept):
          pos_ngram_str = ' '.join(pos_ngram)
          text_ngram_str = ' '.join(text_ngram)
          pos_textlist_dict[pos_ngram_str].append(text_ngram_str)

        # update the entities length and kept length
        kept_ids = list( range(start, end) )
        entity_kept_ids = [ id for id in gen_ent_ids if id in kept_ids]
        kept_ent_len += len(entity_kept_ids)
        kept_len += end - start
        num_user_kept += 1

    # simple_check_doc_tokens(gen_doc, gen_tokens, idx, 'generated')
    # simple_check_doc_tokens(fin_doc, fin_tokens, idx, 'finalized')

  print(f'analyzed {idx + 1} user judgements')



  # convert a counter to a list of tuple, changing its keys from tuple to str
  user_kept_pos_ngram_counter_strkey = collections.Counter( dict([(' '.join(k), v ) for k,v in user_kept_pos_ngram_counter.items() ]))
  all_gen_pos_ngram_counter_strkey = collections.Counter( dict([(' '.join(k), v ) for k,v in all_gen_pos_ngram_counter.items() ]))




  # traverse over the pos ngrams user kept, compute their ratio
  print('traverse through the pos ngrams user kept, compute their ratio')
  kept_gen_pos_ngram_ratio_dict = {}
  for k, v in user_kept_pos_ngram_counter_strkey.most_common(50):
    kept_gen_pos_ngram_ratio_dict[k] = user_kept_pos_ngram_counter_strkey[k] / all_gen_pos_ngram_counter_strkey[k]
    # if all_gen_pos_ngram_counter_strkey[k] ==0:
    #   print('error')


  items = kept_gen_pos_ngram_ratio_dict.items()
  sorted_items = sorted(items, key=lambda key_value: key_value[1], reverse=True)
  import csv
  # write the pos ngrams analysis results into google spread sheet
  with open( os.path.join( args.output_dir, f'{N}_grams_pos_{type_interested}.csv'), mode='w') as csvf:
    fieldnames = ['pos pattern',
                  'ratio (NOK / NOG)',
                  'Number of Occurrences in user-Kept text (NOK)',
                  'Number of Occurrences in Generated text (NOG)',
                  'Examples in user-kept text']
    writer = csv.writer(csvf, delimiter='\t')
    writer.writerow(fieldnames)

    for idx in range(min([50, len(sorted_items)])):
      pos, ratio = sorted_items[idx]
      ck = user_kept_pos_ngram_counter_strkey[pos]
      cg = all_gen_pos_ngram_counter_strkey[pos]
      writer.writerow( [pos, ratio, ck, cg, ', '.join(pos_textlist_dict[pos][:3])] )

  print(f'gen len: {gen_len}')
  print(f'gen ent len: {gen_ent_len}')

  print(f'kept len: {kept_len}')
  print(f'kept ent len: {kept_ent_len}')

  print(f'in generated text the percentage of entity is {gen_ent_len / gen_len}')
  print(f'in kept text the percentage of entity is {kept_ent_len / kept_len}')

  print(f'number of user kept edits: {num_user_kept}')
  print(f'average user kept edit len: {kept_len / num_user_kept}')
  print('End of main')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='./judgement_contexts.json')
  parser.add_argument('--output_dir', type=str, default='./ling_analysis')
  parser.add_argument('--num_grams', type=int, default=1)
  args = parser.parse_args()
  main(args)
  print('Done')


