import os
import pickle
import time

import torch
import numpy as np
from collections import Counter
import concurrent.futures

import argparse

import dae_model
from dae_model import DictionaryAutoencoder

from data_analysis_utils import (read_from_json_export_and_organize_by_world, organize_story_by_world, flatten, convert_token_2_ids,
                                tokenize_idtext, extract_textlist_from_world_story_dict, build_reduced_glove_dict)

parser = argparse.ArgumentParser()

parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--full_export_data_path', type=str, default='/data/storium/full_export_data')
parser.add_argument('--splits_info_path', type=str, default='/data/storium/')
parser.add_argument('--glove_data_path', type=str, default='/data/cc_glove')
parser.add_argument('--glove_file_name', type=str, default='glove.840B.300d.txt')
parser.add_argument('--final_data_path', type=str, default='./final_pooled')
parser.add_argument('--type_text', type=str, default='entry_challenge_pooled',\
                    choices=['entry_challenge_pooled', 'entry', 'challenge'])
args = parser.parse_args()


# step 0: get information about world story file info
world_storyfile_dict_path = os.path.join(args.final_data_path, 'world_storyfile_dict.pkl')
if os.path.exists( world_storyfile_dict_path ):
    world_storyfile_dict = pickle.load( open(world_storyfile_dict_path, 'rb'))
else:
    world_storyfile_dict = organize_story_by_world(args.splits_info_path, args.full_export_data_path)
    pickle.dump(world_storyfile_dict, open(world_storyfile_dict_path, 'wb') )



# step 1: get text by story world dictionary
print('\n\n')
text_by_story_world_fname = os.path.join( args.final_data_path, f'text_by_story_world_dict.pkl' )
if os.path.exists( text_by_story_world_fname ):
    text_by_story_world_dict = pickle.load( open(text_by_story_world_fname, 'rb'))
else:
    text_by_story_world_dict = read_from_json_export_and_organize_by_world(args.full_export_data_path, world_storyfile_dict, \
                                                               args.type_text)
    pickle.dump(text_by_story_world_dict, open(text_by_story_world_fname, 'wb') )


# step 2: get a list of id and data tuple (a data tuple is (content, dict of metadata))
uid_text_list_fname = os.path.join( args.final_data_path, 'uid_text_list.pkl')
if os.path.exists(( uid_text_list_fname) ):
    uid_text_list = pickle.load( open(uid_text_list_fname, 'rb') )
else:
    uid_text_list = extract_textlist_from_world_story_dict(text_by_story_world_dict)
    pickle.dump(uid_text_list, open(uid_text_list_fname, 'wb'))



# uid_text_list = uid_text_list[:1000]

# step 3: tokenize, stem and lemmatize the docs
# convert text by story world dict to a list of tuples, each tuple is (txt_id, text, meta-data)
uid_tokens_list_fname = os.path.join( args.final_data_path, 'id_tokens_list.pkl' )
if os.path.exists( uid_tokens_list_fname ):
    uid_tokens_list = pickle.load( open( uid_tokens_list_fname, 'rb'))
else:
    start = time.time()
    r_list= []
    chunk_size = 100
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(tokenize_idtext, uid_text_list, chunksize=chunk_size)
        for result in results:
            r_list.append(result)
    pickle.dump( r_list, open(uid_tokens_list_fname, 'wb') )

    end = time.time()
    print(f'Chunksize = {chunk_size}, The time spent on tokenizing {len(uid_text_list)} examples is {end - start}, ave {(end - start)/len(uid_text_list)} per example')



# Step 4 get unique vocabuaries from tokens_list
print('\n\n')
if os.path.exists(os.path.join(args.final_data_path, 'uniq_vocab_list.pkl')):
    uniq_vocab_list = pickle.load( open( os.path.join(args.final_data_path, 'uniq_vocab_list.pkl'), 'rb'))
    print(f'From the documents, there are {len(uniq_vocab_list)} unique vocabs (loaded from preprocessed)')

else:
    tokens_list = [ item[1] for item in uid_tokens_list if len(item)==2]
    uniq_vocab_list = list(set(list(flatten(tokens_list))))
    pickle.dump(uniq_vocab_list, open( os.path.join(args.final_data_path, 'uniq_vocab_list.pkl') , 'wb'))
    print(f'From the documents, there are {len(uniq_vocab_list)} unique vocabs (obtained by using set)')




# build vocab (emb_matrix, word2id, id2word)
# Step 5 build (limiting to vocab present in dataset) glove based dictionary
print('\n\n')
if os.path.exists( os.path.join(args.final_data_path, 'embedding_matrix.npy') ) and \
    os.path.exists( os.path.join(args.final_data_path, 'id2word_dict.pkl') ) and \
    os.path.exists( os.path.join(args.final_data_path, 'word2id_dict.pkl') ):
    with open(os.path.join(args.final_data_path, 'word2id_dict.pkl'), 'rb') as f:
        word2id_dict = pickle.load(f)

    with open(os.path.join(args.final_data_path, 'id2word_dict.pkl'), 'rb') as f:
        id2word_dict = pickle.load(f)

    embedding_matrix_np = np.load(os.path.join(args.final_data_path, 'embedding_matrix.npy'))

    print('Loaded word2id_dict with length: ')
    print(len(word2id_dict))
    print('Loaded id2word_dict with length: ')
    print(len(id2word_dict))
    print('Loaded pre-trained embedding with shape: ')
    print(embedding_matrix_np.shape)
else:
    print('Building embedding matrix and dictionaries (this may take 30 minutes)')
    build_reduced_glove_dict(uniq_vocab_list, args.glove_data_path, args.glove_file_name, args.final_data_path)

    with open(os.path.join(args.final_data_path, 'word2id_dict.pkl'), 'rb') as f:
        word2id_dict = pickle.load(f)

    with open(os.path.join(args.final_data_path, 'id2word_dict.pkl'), 'rb') as f:
        id2word_dict = pickle.load(f)

    embedding_matrix_np = np.load(os.path.join(args.final_data_path, 'embedding_matrix.npy'))

    print('Loaded word2id_dict with length: ')
    print(len(word2id_dict))
    print('Loaded id2word_dict with length: ')
    print(len(id2word_dict))
    print('Loaded pre-trained embedding with shape: ')
    print(embedding_matrix_np.shape)

# uid_tokens_list = uid_tokens_list[:1000]
# Step 6 create text ids from text tokens
text_ids_fname = os.path.join(args.final_data_path, 'uid_text_ids.npy')
if os.path.exists( text_ids_fname ):
    uid_text_ids = pickle.load( open( text_ids_fname, 'rb') )
    print(f'Load previously converted token ids with number of data instances = {len(uid_text_ids)}')
else:
    uid_text_ids = convert_token_2_ids(word2id_dict, uid_tokens_list)
    print(f'Length of the text_ids list = {len(uid_text_ids)}')
    pickle.dump(uid_text_ids,  open( text_ids_fname, 'wb') )
    print(f'Save word ids converted from tokens with number of data points = {len(uid_text_ids)}')



# step 7 rank tokens based on frequency
if os.path.exists( os.path.join( args.final_data_path, 'freq_result.pkl') ):
    with open(os.path.join(args.final_data_path, 'freq_result.pkl'), 'rb') as f:
        freq_result = pickle.load(f)
else:
    text_ids = [item[1] for item in uid_text_ids if len(item) == 2]
    text_ids_1d = list(flatten(text_ids))
    freq_result = Counter(text_ids_1d)
    with open( os.path.join( args.final_data_path, 'freq_result.pkl'), 'wb') as f:
        pickle.dump(freq_result, f)


# Step 8 rank tokens based on story occurences
'''
story_tokens_dict = {} (default set?)
for item in uid_text_id:
    get the uid, 
    get the story_fname
    text_ids = uid_text_ids[uid] #
    story_tokens[story_fname].add(text_ids) # add into a set
    
for story in story_tokens:
    get uniq_tokens in story
    token_occ_by_story[uniq_tokens] += 1 # https://docs.scipy.org/doc/numpy/user/basics.indexing.html#assigning-values-to-indexed-arrays
'''
if os.path.exists( os.path.join( args.final_data_path, 'token_story_occurence.pkl') ):
    with open(os.path.join(args.final_data_path, 'token_story_occurence.pkl'), 'rb') as f:
        token_story_occurence = pickle.load(f)
else:

    uid_text_meta_dict = dict(uid_text_list)
    from collections import defaultdict
    story_tokens_dict = defaultdict(set)

    for item in uid_text_ids:
        if len(item) ==2:
            uid, text_ids = item[0], item[1]
            curr_game_pid = uid_text_meta_dict[uid][1]['game_pid']
            for id in text_ids:
                story_tokens_dict[curr_game_pid].add(id)

    token_story_occurence = np.zeros( (len(id2word_dict, )) )
    for k, v in story_tokens_dict.items():
        v = list(v)
        token_story_occurence[v] += 1
    pickle.dump( token_story_occurence,  open(os.path.join(args.final_data_path, 'token_story_occurence.pkl'), 'wb'))


# Step 9 :
# random.shuffle(uid_text_ids)
freq_most_common = freq_result.most_common()
freq_reverse = list(reversed(freq_most_common))
to_be_removed = []
for (wordid, count) in freq_reverse:
    if count < args.freq_threshold:
        to_be_removed.append(wordid)
low_freq_cnt = len(to_be_removed)
print(f'In terms of frequency, {low_freq_cnt} words are removed with threshold at {args.freq_threshold}')


for (wordid, cnt) in enumerate(token_story_occurence):
    if wordid not in to_be_removed and cnt <= args.occurrence_threshold:
        to_be_removed.append(wordid)
print(f'In terms of occurrences in stories, {len(to_be_removed) - low_freq_cnt} words are removed with threshold at {args.occurrence_threshold}')
with open( os.path.join(args.final_data_path, 'to_be_removed.pkl'), 'wb' ) as f:
    pickle.dump( to_be_removed,  f)


to_be_removed = list( set( to_be_removed ) )




print('Done')
