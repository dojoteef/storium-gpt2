'''

    We would like to analyze the narrative paths in each world by looking at topic transitions in each world

    refactor to make inference faster

    filter vocab

'''



import sys

print(sys.path)
# import matplotlib
# import matplotlib.pyplot as plt
import random
import torch
from torch import nn, optim
from torch.autograd import Function
import numpy as np
import os
import pickle
import data_analysis_utils
from data_analysis_utils import (prepare_text_for_lda, build_reduced_glove_dict, convert_token_2_ids, text_to_topic,
                                 compute_topic_transition_matrix, filtere_off_unk_topics)
from collections import Counter
import dae_model
from dae_model import DictionaryAutoencoder
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--data_path', type=str, default='./final_pooled_backup')
parser.add_argument('--model_name', type=str, default='no_filter_train_30_10.pt')
parser.add_argument('--topic_path', type=str, default='./final_pooled_backup')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--config', type=str, default='entry_by_character')
parser.add_argument('--freq_threshold', type=int, default=30)
parser.add_argument('--occurrence_threshold', type=int, default=10)

args = parser.parse_args()

device = f'cuda:{str(args.device_id)}'
config = args.config


# fix random seed for python, numpy and torch to ensure reproducibility
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


# load original data info
with open( os.path.join(args.data_path, 'text_by_story_world_dict.pkl'), 'rb') as f:
    text_by_story_world_dict = pickle.load(f)
world_counter = Counter()
for world, stories in text_by_story_world_dict.items():
    world_counter.update({world: len(stories)})
print(f'There are {len(text_by_story_world_dict)} worlds in total')

# load uid information
uid_text_list_fname = os.path.join( args.data_path, 'uid_text_list.pkl')
if os.path.exists(( uid_text_list_fname) ):
    uid_text_list = pickle.load( open(uid_text_list_fname, 'rb') )
uid_info_tuple_list = [ (tuple_item[1][0], tuple_item[0]) for tuple_item in uid_text_list ]
info_uid_tuple_dict = dict(uid_info_tuple_list)

uid_by_story_world_dict_path = os.path.join(args.data_path, 'uid_by_story_world_dict.pkl')
if os.path.exists( uid_by_story_world_dict_path):
    uid_by_story_world_dict = pickle.load( open( uid_by_story_world_dict_path, 'rb'))
else:
    # establish the link between text_by_story_world_dict and uid
    uid_by_story_world_dict = text_by_story_world_dict
    for world, stories in uid_by_story_world_dict.items():
        world_counter.update({world: len(stories)})

        for story_name, info_meta_tupe_list in stories.items():
            for idx, info_meta_tupe in enumerate(info_meta_tupe_list):
                info = info_meta_tupe[0]
                # if info in info_uid_tuple_dict.keys():
                uid = info_uid_tuple_dict[info]
                info_meta_tupe_list[idx] = (info, info_meta_tupe[1], uid)
    pickle.dump(uid_by_story_world_dict, open( uid_by_story_world_dict_path, 'wb' ))




text_ids_fname = os.path.join(args.data_path, 'uid_text_ids.npy')
if os.path.exists( text_ids_fname ):
    uid_text_ids = pickle.load( open( text_ids_fname, 'rb') )
    print(f'Load previously converted token ids with number of data instances = {len(uid_text_ids)}')





embedding_matrix_np = np.load(os.path.join(args.data_path, 'embedding_matrix.npy'))
with open(os.path.join(args.data_path, 'word2id_dict.pkl'), 'rb') as f:
    word2id_dict = pickle.load(f)
with open(os.path.join(args.data_path, 'id2word_dict.pkl'), 'rb') as f:
    id2word_dict = pickle.load(f)



with open(os.path.join(args.data_path, args.model_name), 'rb') as f:
    model = torch.load(f)
model.device = device
model.to(device)
model.eval()



uid_topic_list_path = os.path.join(args.data_path, 'uid_topic_list')
if not os.path.exists( uid_topic_list_path ):
    uid_input_vector_list = []
    for i in range( len( uid_text_ids) ): # loop over each challenge
        uid_text_tuple = uid_text_ids[i]
        if len(uid_text_tuple) ==2:
            uid, sent = uid_text_tuple
            vectors = [ embedding_matrix_np[word_id, :] for word_id in sent]

            if len(vectors) > 0:
                vector_mean = np.mean( vectors, axis=0)
                uid_input_vector_list.append( (uid,vector_mean) )

    uid_list, vector_list = zip( *uid_input_vector_list )

    t0 = time.time()
    topic_pred_list = text_to_topic(vector_list, model, device, batch_size=400)
    uid_topic_list = zip(uid_list, topic_pred_list)
    t1 = time.time()
    print(f'Time taken to assign topics to all text is {t1 - t0}')

    with open( os.path.join(args.data_path, 'uid_topic_list'), 'wb' ) as f:
        pickle.dump(uid_topic_list, f)

else:
    uid_topic_list = pickle.load( open(uid_topic_list_path, 'rb'))
uid_topic_dict = dict(uid_topic_list)


num_topics = 50

t0 = time.time()
identifier = 'role' # or 'game_pid'
type_text= 'entry'
topic_transition_matrix_by_world_dict = {}

num_all, num_key_not_found, num_not_entry, num_counted = 0, 0, 0, 0

for idx, world_tuple in enumerate(world_counter.most_common()):
    if idx == 0: # ignore the None world
        continue
    world_name = world_tuple[0]
    stories = uid_by_story_world_dict[world_name]
    world_topics_transition_matrix = np.zeros((num_topics, num_topics))

    identifier_topic_list_dict = {}

    for story_fname, inf_meta_uid_triples_list in stories.items(): # each story contains a list of text
        topic_list_per_identifier_dict = {}
        for inf_meta_uid_triple in inf_meta_uid_triples_list:
            _, meta, uid = inf_meta_uid_triple
            id_key = meta[identifier]

            num_all += 1
            if uid not in uid_topic_dict.keys():
                # print('The uid is not in the uid_topic_dict keys')
                num_key_not_found+=1

            if meta['type'] != type_text:
                # print('The type of text is not entry')
                num_not_entry += 1

            if uid in uid_topic_dict.keys() and meta['type'] == type_text:
                if uid_topic_dict[uid] == 25: # hardcoded for the "stop word topic"
                    continue
                num_counted += 1
                if id_key in identifier_topic_list_dict.keys():
                    identifier_topic_list_dict[id_key].append( uid_topic_dict[uid] )
                else:
                    identifier_topic_list_dict[id_key] = [ uid_topic_dict[uid] ]

    for id_key, topic_list in identifier_topic_list_dict.items():
        world_topics_transition_matrix += compute_topic_transition_matrix(topic_list, num_topics)

    topic_transition_matrix_by_world_dict[world_name] = world_topics_transition_matrix

    # if idx >=7:
    #     break
    print(f'Done with {idx+1}  worlds')

print(f'Number over all: {num_all}')
print(f'Number key not found: {num_key_not_found}')
print(f'Number type text not entry: {num_not_entry}')
print(f'Number counted: {num_counted}')


with open(os.path.join( args.data_path, f'topic_transition_matrix_by_world_dict_{identifier}.pkl'), 'wb') as f:
    pickle.dump(topic_transition_matrix_by_world_dict, f)


with open(os.path.join( args.data_path, f'topic_transition_matrix_by_world_dict_{identifier}.pkl'), 'rb') as f:
    topic_transition_matrix_by_world_dict = pickle.load(f)

t1 = time.time()

print(f'Total time taken to compute topic transition matrix by worlds is {t1 - t0}')
print('\n\n')
# for world_name, topic_transition_matrix in topic_transition_matrix_by_world_dict.items():
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(topic_transition_matrix)
#
#     # We want to show all ticks...
#     ax.set_xticks(np.arange(num_topics))
#     ax.set_yticks(np.arange(num_topics))
#
#     # # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=90, ha="right", size=9,
#              rotation_mode="anchor")
#
#
#     ax.set_title(f"Inter-topic transition in {world_name}")
#     fig.tight_layout()
#     plt.savefig( os.path.join(f'./vis/entry/{ "_".join(world_name.split())}'))


with open( os.path.join(args.topic_path, 'topics.txt'), 'r') as f:
    lines = f.readlines()

topics_summarized = []
idx_filter_off = []
for idx, line in enumerate( lines ):
    line_split = line.strip().split(': ')[1]
    topics_summarized.append(line_split)
    if line_split[:3] == 'unk':
        idx_filter_off.append(idx)

#
# for starting in range(num_topics):
#     print('\n\n')
#     print( '=' * 100)
#
#     starting_topic = topics_summarized[starting]
#     if starting_topic[:3]== 'unk':
#         continue
#     print(f'Starting from topic {starting} {topics_summarized[starting]}: ')
#     for idx, (world_name, _) in enumerate(world_counter.most_common()):
#         if idx == 0:
#             continue
#         topic_transition_matrix = topic_transition_matrix_by_world_dict[world_name]
#         topic_transition_matrix = filtere_off_unk_topics(topic_transition_matrix, idx_filter_off)
#
#
#         # traverse through topic transition matrix
#         track = []
#
#         next = starting
#         track.append(next)
#         while len(track) < 4:
#             all_candidates = topic_transition_matrix[next]
#             desc_order = np.argsort(all_candidates)
#             if next == desc_order[-1]:
#                 next = desc_order[-2]
#             else:
#                 next = desc_order[-1] # -2 because the highest probability other than the diagonal
#             track.append(next)
#
#         track_str = []
#         for i in track:
#             track_str.append(topics_summarized[i])
#         # print(f'In the world {world_name}, starting from {topics_summarized[starting]}:')
#         print(f'[{world_name}]: ' + ' --> '.join(track_str))
#         print()
#
#         if idx >= 7:
#             break
#     print('\n\n')

#
for starting in range(num_topics):
    print('\n\n')
    print( '=' * 100)

    starting_topic = topics_summarized[starting]
    if starting_topic[:3]== 'unk':
        continue
    print(f'Starting from topic {starting} {topics_summarized[starting]}: ')
    for idx, (world_name, _) in enumerate(world_counter.most_common()):
        if idx == 0:
            continue
        topic_transition_matrix = topic_transition_matrix_by_world_dict[world_name]
        topic_transition_matrix = filtere_off_unk_topics(topic_transition_matrix, idx_filter_off)

        all_candidates = topic_transition_matrix[starting]
        desc_order = np.argsort(all_candidates)

        top_3 = desc_order[-3:]

        if starting in top_3:
            indx_of_starting = np.where( top_3 == starting)[0][0]
            top_2 = np.delete(top_3, indx_of_starting)
        else:
            top_2 = top_3[-2:]

        top_2_words = [topics_summarized[i] for i in top_2 ]
        top_2_str = ' ' + ' | '.join(top_2_words) + ' '
        print(f'[{world_name}]: '+ f' {topics_summarized[starting]} '  + ' --> ' + top_2_str)

        if idx >= 7:
            break

print('Done')