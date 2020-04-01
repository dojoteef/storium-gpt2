'''

    We would like to analyze the narrative paths in each world by looking at topic transitions in each world

    refactor to make inference faster

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
from data_analysis_utils import (prepare_text_for_lda, read_from_json_export, tokenize_doc_list, build_reduced_glove_dict,
convert_token_2_ids, read_from_json_export_by_story, text_to_topic, compute_topic_transition_matrix, filtere_off_unk_topics)
from collections import Counter
import dae_model
from dae_model import DictionaryAutoencoder
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--stored_data_path', type=str, default='./pooled_entry_challenge_data')
parser.add_argument('--model_name', type=str, default='model.pt')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--topic_path', type=str, default='./topics_info')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--config', type=str, default='entry_by_character')
args = parser.parse_args()

device = f'cuda:{str(args.device)}'
config = args.config


# fix random seed for python, numpy and torch to ensure reproducibility
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

with open( os.path.join('./entry_data', 'text_by_story_world_dict.pkl'), 'rb') as f:
    text_by_story_world_dict = pickle.load(f)


if args.config == 'challenge_by_story':
    with open('./stored_data/text_by_story_world_dict.pkl', 'rb') as f:
        text_by_story_world_dict = pickle.load(f)

world_counter = Counter()
print(f'There are {len(text_by_story_world_dict)} worlds in total')
for world, stories in text_by_story_world_dict.items():
    world_counter.update({world: len(stories)})



with open(os.path.join(args.stored_data_path, args.model_name), 'rb') as f:
    model = torch.load(f, map_location=device)  # put the model on specified device

model.to(device)  # simply load without parallelism
model.eval()



embedding_matrix_np = np.load(os.path.join(args.stored_data_path, 'embedding_matrix.npy'))
with open(os.path.join(args.stored_data_path, 'word2id_dict.pkl'), 'rb') as f:
    word2id_dict = pickle.load(f)


print('Topics with nearest neighbour decoding: ')
model.interpret_dictionary()
print('=' * 70)
print('Topics with probability argmax: ')
model.rank_vocab_for_topics(word_embedding_matrix=embedding_matrix_np)




num_topics = 50


topic_transition_matrix_by_world_dict = {}

t0 = time.time()

# for idx, world_tuple in enumerate(world_counter.most_common()):
#     if idx == 0: # ignore the None world
#         continue
#     world_name = world_tuple[0]
#     stories = text_by_story_world_dict[world_name]
#     world_topics_transition_matrix = np.zeros((num_topics, num_topics))
#     for story_fname, text_list in stories.items(): # each story contains a list of text
#         if config == 'entry_by_character':
#             roles_in_story = [ role for sent, role in text_list if sent is not None]
#             for role_cur in roles_in_story:
#                 text_list_of_role = [sent for sent, role in text_list if role == role_cur and sent is not None]
#                 if len(text_list_of_role) > 0: # a list of text gets mapped to a list of topic ids
#                     story_topic_pred_list = text_to_topic(text_list_of_role, word2id_dict, embedding_matrix_np, model, device, 200)
#                     world_topics_transition_matrix += compute_topic_transition_matrix(story_topic_pred_list, num_topics)
#         if config == 'entry_by_story':
#             text_list = [ sent for sent, role in text_list if sent is not None]
#             if len(text_list) > 0: # a list of text gets mapped to a list of topic ids
#                 story_topic_pred_list = text_to_topic(text_list, word2id_dict, embedding_matrix_np, model, device, 200)
#                 world_topics_transition_matrix += compute_topic_transition_matrix(story_topic_pred_list, num_topics)
#         if config == 'challenge_by_story':
#             text_list = [ sent for sent, role in text_list if sent is not None]
#             if len(text_list) > 0: # a list of text gets mapped to a list of topic ids
#                 story_topic_pred_list = text_to_topic(text_list, word2id_dict, embedding_matrix_np, model, device, 200)
#                 world_topics_transition_matrix += compute_topic_transition_matrix(story_topic_pred_list, num_topics)
#     topic_transition_matrix_by_world_dict[world_name] = world_topics_transition_matrix
#
#     if idx >=7:
#         break
#
#     print(f'Done with {idx+1} / 7 worlds')
# with open(os.path.join( args.stored_data_path, f'topic_transition_matrix_by_world_dict_{config}.pkl'), 'wb') as f:
#     pickle.dump(topic_transition_matrix_by_world_dict, f)


with open(os.path.join( args.stored_data_path, f'topic_transition_matrix_by_world_dict_{config}.pkl'), 'rb') as f:
    topic_transition_matrix_by_world_dict = pickle.load(f)


t1 = time.time()

print(f'Total time taken for text to topic is:')
print(t1 - t0)
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


with open( os.path.join(args.topic_path, 'topics_pooled.txt'), 'r') as f:
    lines = f.readlines()

topics_summarized = []
idx_filter_off = []
for idx, line in enumerate( lines ):
    line_split = line.strip().split(': ')[1]
    topics_summarized.append(line_split)
    if line_split[:3] == 'unk':
        idx_filter_off.append(idx)


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
#         while len(track) < 6:
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