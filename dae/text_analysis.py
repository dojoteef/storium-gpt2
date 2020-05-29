'''

    Analyze the linguistic

'''

import os, pickle, random
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
from spacy.gold import align


def simple_check_doc_tokens(doc, tokens, idx, type):
  if not ( len(doc) == len(tokens) ):
    print(f'type input: {type}')
    print(f'number: {idx}')
    print(f'num spacy tokens: {len(doc)}')
    print(f'num rouge tokens: {len(tokens)}')
    print()

# def custom_tokenizer(text):
#     # tokens = []
#     #
#     # # your existing code to fill the list with tokens
#     #
#     # # replace this line:
#     # return tokens
#     tokens = text.split()
#     # with this:
#     return Doc(nlp.vocab, tokens)
# nlp.tokenizer = custom_tokenizer

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


def process_spacy_tokens_for_align(gen_spacy_tokens_unprep):
  gen_spacy_tokens = [token.lower() for token in gen_spacy_tokens_unprep]

  # process gen_tokens to take care of corner case like "'s" vs "s"
  for token_idx, token in enumerate(gen_spacy_tokens):
    if token == "'s":
      gen_spacy_tokens[token_idx] = 's'
    elif token == 'mrs.' or token == 'dr.':
      gen_spacy_tokens[token_idx] = token[:-1]
  return gen_spacy_tokens


def obtain_predefined_entities(story):
  char_list = story['characters']
  char_name_lc_list = [ entry['name'].lower() for entry in char_list]
  return char_name_lc_list

def contained_in(entity, predefined_entity_list):
  for entry in predefined_entity_list:
    if entity in entry:
      return True
  return False


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
  interest_len = 0
  interest_ent_len = 0
  num_user_interest = 0
  num_interested = 0
  num_corner_case = 0

  num_uniq_gen_entities = 0
  num_uniq_interest_entities = 0
  num_uniq_gen_predefined_entities = 0
  num_uniq_interest_predefined_entities = 0

  for data_idx, item in enumerate(data):
    # copy the existing content, a dict of 4 fields
    new_dict = item
    story = new_dict['story']
    predefined_characters = obtain_predefined_entities(story)
    generated = item['generated']['description']
    finalized = item['finalized']['description']
    gen_doc = nlp(generated)
    gen_spacy_tokens_unprep = [ token.text for token in gen_doc]
    gen_spacy_tokens = process_spacy_tokens_for_align(gen_spacy_tokens_unprep)

    gen_prep = rouge._preprocess_summary_as_a_whole(generated)
    gen_prep_zero = gen_prep[0]
    gen_rouge_tokens = gen_prep_zero.split()

    cost, s2r, r2s, s2r_multi, r2s_multi = align(gen_spacy_tokens, gen_rouge_tokens)

    new_dict['diffs'] = get_diff_score(generated, finalized)
    judgement_diffs.append(new_dict)

    gen_pos = [ token.pos_ for token in gen_doc]
    all_gen_pos_ngram_counter.update(ngrams(gen_pos, N))

    ents = gen_doc.ents
    gen_ent_ids = []
    for ent in ents:
      ent_ids = list(range(ent.start, ent.end))
      gen_ent_ids.extend(ent_ids)
    gen_len += len(gen_pos)
    gen_ent_len += len(gen_ent_ids)

    gen_ent_str_list = []
    gen_ent_str_predefined_list = []
    for ent in ents: # number of uniq entities
      ent_lc = ent.text.lower()
      gen_ent_str_list.append(ent_lc)
      if contained_in(ent_lc, predefined_characters):
        gen_ent_str_predefined_list.append(ent_lc)

    print('Generated entities')
    print( ' | '.join(gen_ent_str_list))
    print('Predefined characters: ')
    print(' | '.join(predefined_characters))
    print('Generated entities that are kept in the predefined character list: ')
    print(' | '.join(gen_ent_str_predefined_list))
    print('\n\n')

    gen_ent_str_int_set = set()
    gen_ent_str_int_prefefined_set = set()


    for diff_results in new_dict['diffs']:
      if diff_results['type'] == args.type_interested:
        num_user_interest += 1

        [start_r, end_r] = diff_results['span_hypothesis']
        content = diff_results['content']
        content_rouge = gen_rouge_tokens[start_r:end_r]

        if end_r == len(r2s): # this is a corner case because end_r is not actually the ending index of a span but that + 1
          start_s, end_s = r2s[start_r], r2s[end_r-1] + 1 # bug causing
        else:
          start_s, end_s = r2s[start_r], r2s[end_r] # bug causing

        content_spacy = gen_spacy_tokens[start_s:end_s]

        if content_spacy == []:
          # print('handle corner case')
          # print(content)
          # print(content_rouge)
          start_s = end_s - 1
          content_spacy = gen_spacy_tokens[start_s:end_s]
          # print(content_spacy)
          # print('\n\n')
          # num_corner_case += 1

        # print( ' | '.join(content_spacy))



        cur_ngrams_kept = list( ngrams(content_spacy, N) ) # a list of n grams of actual text
        pos_tokens = gen_pos[start_s:end_s]
        cur_ngrams_pos = list( ngrams(pos_tokens, N) ) # a list of n grams of pos
        user_kept_pos_ngram_counter.update(cur_ngrams_pos) # update the count of POS ngrams

        assert len(content_spacy) == len(pos_tokens)
        assert len(cur_ngrams_kept) == len(cur_ngrams_pos)

        # update the dictionary (str, list): (pos_ngram, list of actual ngrams as examples)
        for pos_ngram, text_ngram in zip(cur_ngrams_pos, cur_ngrams_kept):
          pos_ngram_str = ' '.join(pos_ngram)
          text_ngram_str = ' '.join(text_ngram)
          pos_textlist_dict[pos_ngram_str].append(text_ngram_str)

        interest_ids = list( range(start_s, end_s) )
        entity_interest_ids = [ id for id in interest_ids if id in gen_ent_ids]
        interest_ent_len += len(entity_interest_ids)
        interest_len += end_s - start_s

        # check whether the generated entities appear in diff
        for entity in gen_ent_str_list:
          if entity in ' '.join(content_spacy):
            gen_ent_str_int_set.add(entity)

            # check whether an entity in this type of text is present in the set of predefined characters
            if contained_in(entity, predefined_characters):
              gen_ent_str_int_prefefined_set.add(entity)
              print(entity)
            # else:
            #   # print(entity)

    num_uniq_gen_entities += len(gen_ent_str_list)
    num_uniq_interest_entities += len(gen_ent_str_int_set)
    num_uniq_gen_predefined_entities += len(gen_ent_str_predefined_list)
    num_uniq_interest_predefined_entities += len(gen_ent_str_int_prefefined_set)
    # print(gen_ent_str_list)
    # print(gen_ent_str_int_set)
    # print()



  print(f'analyzed {data_idx + 1} user judgements')

  print(f'Out of {num_interested} interested diff')
  print(f'there are {num_corner_case} corner cases')

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
  with open( os.path.join( args.output_dir, f'{N}_grams_pos_{args.type_interested}.csv'), mode='w') as csvf:
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
      writer.writerow( [pos, ratio, ck, cg, '  ||  '.join( random.sample(pos_textlist_dict[pos], 5)  )] )
  print('Saved: ')
  print(os.path.join( args.output_dir, f'{N}_grams_pos_{args.type_interested}.csv'))
  print(f'gen len: {gen_len}')
  print(f'gen ent len: {gen_ent_len}')

  print(f'{args.type_interested} len: {interest_len}')
  print(f'{args.type_interested} ent len: {interest_ent_len}')

  print(f'in generated text the percentage of entity is {gen_ent_len / gen_len}')
  print(f'in {args.type_interested} text the percentage of entity is {interest_ent_len / interest_len}')

  print(f'number of user {args.type_interested} edits: {num_user_interest}')
  print(f'average user {args.type_interested} edit len: {interest_len / num_user_interest}')

  print(f'Number of entities generated: {num_uniq_gen_entities}')
  print(f'Number of entities generated that are from predefined character list: {num_uniq_gen_predefined_entities}')
  print(f'Number of entities in {args.type_interested}: {num_uniq_interest_entities}')
  print(f'Number of entities in {args.type_interested} that are from predefined character list:{num_uniq_interest_predefined_entities}')

  print('End of main')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='./judgement_contexts_bl.json')
  parser.add_argument('--output_dir', type=str, default='./ling_analysis')
  parser.add_argument('--num_grams', type=int, default=1)
  parser.add_argument('--type_interested', type=str, default='user-kept')
  args = parser.parse_args()
  main(args)
  print('Done')


