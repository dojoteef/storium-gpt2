'''

    After computing the topic_transition_matrix_by_world_dict in world_path.py, we can visualize them
    1. pie chart


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
                                 compute_topic_transition_matrix, filtere_off_unk_topics, generate_latex_table)
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
identifier = 'role'
type_text = 'entry'
with open(os.path.join( args.data_path, f'topic_transition_matrix_by_world_dict_{identifier}.pkl'), 'rb') as f:
    topic_transition_matrix_by_world_dict = pickle.load(f)



# load original data info
with open( os.path.join(args.data_path, 'text_by_story_world_dict.pkl'), 'rb') as f:
    text_by_story_world_dict = pickle.load(f)
world_counter = Counter()
for world, stories in text_by_story_world_dict.items():
    world_counter.update({world: len(stories)})
uid_by_story_world_dict_path = os.path.join(args.data_path, 'uid_by_story_world_dict.pkl')
if os.path.exists( uid_by_story_world_dict_path):
    uid_by_story_world_dict = pickle.load( open( uid_by_story_world_dict_path, 'rb'))
else:
    print('Please run the world_path')
uid_topic_list_path = os.path.join(args.data_path, 'uid_topic_list')
uid_topic_list = pickle.load( open(uid_topic_list_path, 'rb'))
uid_topic_dict = dict(uid_topic_list)


num_all, num_key_not_found, num_not_entry, num_counted = 0, 0, 0, 0
num_topics = 50


world_topic_freq_list_dict = {}
overall_freq = np.zeros( (num_topics,) )
for idx, world_tuple in enumerate(world_counter.most_common()):
    # if idx == 0: # ignore the None world
    #     continue
    world_name = world_tuple[0]
    stories = uid_by_story_world_dict[world_name]
    world_topics_transition_matrix = np.zeros((num_topics, num_topics))

    world_topic_freq_list = np.zeros( (num_topics, ) )

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
                world_topic_freq_list[uid_topic_dict[uid]] += 1

    world_topic_freq_list_dict[world_name] = world_topic_freq_list
    overall_freq += world_topic_freq_list
    # print(f'Done with {idx+1}  worlds')





with open( os.path.join(args.topic_path, 'topics.txt'), 'r') as f:
    lines = f.readlines()

topics_summarized = []
idx_filter_off = []
for idx, line in enumerate( lines ):
    line_split = line.strip().split(': ')[1]
    topics_summarized.append(line_split)
    if line_split[:3] == 'unk':
        idx_filter_off.append(idx)


print()
def get_top_topics(overall_freq, topics_summarized, n_words, cutoff, threshold):
    overall_norm = overall_freq / np.sum(overall_freq)
    content = []
    freq_list = []
    for idx, ind in enumerate( reversed( np.argsort(overall_norm) ) ):
        if ind == 25: ## the stop word topic
            continue
        topic_words_str = ', '.join( topics_summarized[ind].split(', ')[:n_words] )
        freq_percentage = np.around(overall_norm[ind] * 100, 2)
        # print(f' freq: { freq_percentage } % {topic_words_str }')
        # print(ind)
        if freq_percentage < threshold:
            break

        freq_list.append(freq_percentage)
        content.append( [topic_words_str] )
        # if len(freq_list) == cutoff:
        #     break

    return freq_list, content



def get_relative_important_topics(given_freq, overall_freq, topics_summarized, n_words, cutoff, threshold):
    overall_freq_norm = overall_freq / np.sum(overall_freq)
    given_freq_norm = given_freq / np.sum(given_freq)
    relative_impt = given_freq_norm / (overall_freq_norm + 0.000005)
    content = []
    freq_list = []
    impt_list = []
    for idx, ind in enumerate( reversed( np.argsort(relative_impt) ) ):
        if ind == 25: ## the stop word topic
            continue
        topic_words_str = ', '.join( topics_summarized[ind].split(', ')[:n_words] )
        freq_percentage = np.around( given_freq_norm[ind] * 100, 2)
        # print(f' freq: { freq_percentage } % {topic_words_str }')
        # print(ind)
        if freq_percentage < threshold:
            continue

        impt_list.append(relative_impt[ind])
        freq_list.append(freq_percentage)
        content.append( [topic_words_str] )
        # if len(freq_list) == cutoff:
        #     break

    return impt_list, freq_list, content




cutoff = 20
n_words = 6
threshold = 1
# plot all tables of relative importance
world_row_names = []
topic_row_list = []
for idx, world_tuple in enumerate(world_counter.most_common()):
    if idx == 0: # ignore the None world
        continue
    world_name = world_tuple[0]
    print(world_name)
    # get_top_topics(world_topic_freq_list_dict[world_name], topics_summarized, 20)
    impt_list, freq_list, content= get_relative_important_topics(world_topic_freq_list_dict[world_name], overall_freq, topics_summarized, n_words, cutoff, threshold)
    # generate_latex_table(content=content, Sci_Notn=False, row_names = freq_list, col_names = ['topic words'], row_header='freq(\%)', caption=world_name)

    # print(world_name + ': ')
    # print(f'relative importance = {impt_list[0]}, freq = {freq_list[0]}, topic words: {content[0]}')

    world_row_names.append(world_name)
    topic_row_list.append( content[0] )

    print()
    if idx == 50:
        break

    # print("\\\\")
caption = 'CAPTION HERE'
generate_latex_table(content=topic_row_list, Sci_Notn=False, row_names = world_row_names, col_names = ['topic words'], row_header='worlds', caption=caption)




topics_2w = []
idx_filter_off = []
for idx, line in enumerate( lines ):
    line_split = line.strip().split(': ')[1]
    two_words = line_split.split(', ')[:2]
    topics_2w.append(' '.join(two_words))


def get_top_2(starting, topic_transition_matrix):
    all_candidates = topic_transition_matrix[starting]
    desc_order = np.argsort(all_candidates)

    top_3 = desc_order[-3:]

    if starting in top_3:
        indx_of_starting = np.where(top_3 == starting)[0][0]
        top_2 = np.delete(top_3, indx_of_starting)
    else:
        top_2 = top_3[-2:]
    return top_2



num_topics = 50
for starting in range(num_topics):
    if starting != 20:
        continue

    print('\n\n')
    print( '=' * 100)

    starting_topic = topics_2w[starting]

    print(f'Starting from topic {starting} {topics_summarized[starting]}: ')
    for idx, (world_name, _) in enumerate(world_counter.most_common()):
        print(f'[{world_name}]: ')
        if idx == 0:
            continue
        topic_transition_matrix = topic_transition_matrix_by_world_dict[world_name]
        one_and_two = get_top_2(starting, topic_transition_matrix)

        three_and_four = get_top_2(one_and_two[0], topic_transition_matrix)

        five_and_six = get_top_2(one_and_two[1], topic_transition_matrix)

        result = [starting, one_and_two[0], one_and_two[1], three_and_four[0], three_and_four[1], five_and_six[0], five_and_six[1] ]

        print(f'            {result[0]}              ')
        print(f'            {result[1]}              {result[2]}')
        print(f'            {result[3]}              {result[4]}            {result[5]}              {result[6]}')

        print(f'            {topics_2w[result[0]]}              ')
        print(f'            {topics_2w[result[1]]}              {topics_2w[result[2]]}')
        print(f'            {topics_2w[result[3]]}              {topics_2w[result[4]]}            {topics_2w[result[5]]}              {topics_2w[result[6]]}')


        print('\n\n')
        if idx >= 7:
            break



print('Done')
