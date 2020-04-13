''''''
import sys

print('Current path: ')
print(sys.path)


import torch
import random
from torch import nn, optim
from torch.autograd import Function
import numpy as np
import os
import pickle
import data_analysis_utils
from data_analysis_utils import (prepare_text_for_lda, flatten)
from langdetect import detect

import dae_model
from dae_model import DictionaryAutoencoder
import time
import argparse
from collections import Counter
from model_utils import run_epoch
import types





parser = argparse.ArgumentParser()

# parameters to tune [can be changed]
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--dropout_rate', type=float, default=0.2)
parser.add_argument('--d_hidden', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--use_norm_we', help='normalizing the glove embedding',action='store_true')
parser.add_argument('--use_recon_loss', help='whether we use the reconstruction loss',action='store_true')
parser.add_argument('--pred_world_label', help='whether we use the reconstruction loss',action='store_true')
parser.add_argument('--focus_on_world', type=str, default='')


parser.add_argument('--freq_threshold', type=int, default=25)
parser.add_argument('--occurrence_threshold', type=int, default=5)


parser.add_argument('--triplet_loss_margin', type=float, default=1.0)

parser.add_argument('--ortho_weight', type=float, default=1e-4)
parser.add_argument('--world_clas_weight', type=float, default=0.0)
parser.add_argument('--triplet_loss_weight', type=float, default=1.0)
parser.add_argument('--num_topics', type=int, default=50)
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--num_negative_samples', type=int, default=5)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--num_world', type=int, default=59) # 59 worlds, including the none
parser.add_argument('--ignore_none_world', action='store_true')
parser.add_argument('--freq_filter_train', action='store_true')

parser.add_argument('--bert_data_path', type=str, default='./bert_data')
parser.add_argument('--final_data_path', type=str, default='./final_pooled')
parser.add_argument('--splits_info_path', type=str, default='/data/datasets/storium')
parser.add_argument('--full_export_data_path', type=str, default='/data/datasets/storium/full_export_data')
parser.add_argument('--saved_model_path', type=str, default='')
parser.add_argument('--stored_data_path', type=str, default='./pooled_entry_challenge_data')
parser.add_argument('--type_text', type=str, default='entry_challenge_pooled')





args = parser.parse_args()


# fix random seed for python, numpy and torch to ensure reproducibility
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)



device = torch.device("cuda:" + str(args.device_id))
args.device = device
print('\n\n')
print('The hyperparameters: ')
print(args)


# step 0: get information about world story file info
world_storyfile_dict_path = os.path.join(args.final_data_path, 'world_storyfile_dict.pkl')
if os.path.exists( world_storyfile_dict_path ):
    world_storyfile_dict = pickle.load( open(world_storyfile_dict_path, 'rb'))




# step 1: get text by story world dictionary
print('\n\n')
text_by_story_world_fname = os.path.join( args.final_data_path, f'text_by_story_world_dict.pkl' )
if os.path.exists( text_by_story_world_fname ):
    text_by_story_world_dict = pickle.load( open(text_by_story_world_fname, 'rb'))



# step 2: get a list of id and data tuple (a data tuple is (content, dict of metadata))
uid_text_list_fname = os.path.join( args.final_data_path, 'uid_text_list.pkl')
if os.path.exists(( uid_text_list_fname) ):
    uid_text_list = pickle.load( open(uid_text_list_fname, 'rb') )



# step 3: tokenize, stem and lemmatize the docs
# convert text by story world dict to a list of tuples, each tuple is (txt_id, text, meta-data)
uid_tokens_list_fname = os.path.join( args.final_data_path, 'id_tokens_list.pkl' )
if os.path.exists( uid_tokens_list_fname ):
    uid_tokens_list = pickle.load( open( uid_tokens_list_fname, 'rb'))


# Step 4 get unique vocabuaries from tokens_list
print('\n\n')
if os.path.exists(os.path.join(args.final_data_path, 'uniq_vocab_list.pkl')):
    uniq_vocab_list = pickle.load( open( os.path.join(args.final_data_path, 'uniq_vocab_list.pkl'), 'rb'))
    print(f'From the documents, there are {len(uniq_vocab_list)} unique vocabs (loaded from preprocessed)')




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


# Step 6 create text ids from text tokens
text_ids_fname = os.path.join(args.final_data_path, 'uid_text_ids.npy')
if os.path.exists( text_ids_fname ):
    uid_text_ids = pickle.load( open( text_ids_fname, 'rb') )
    print(f'Load previously converted token ids with number of data instances = {len(uid_text_ids)}')


# random.shuffle(uid_text_ids)
if os.path.exists( os.path.join( args.final_data_path, 'freq_result.pkl') ):
    with open( os.path.join( args.final_data_path, 'freq_result.pkl'), 'rb') as f:
        freq_result = pickle.load(f)

freq_most_common = freq_result.most_common()
freq_reverse = list(reversed(freq_most_common))
to_be_removed = []
for (wordid, count) in freq_reverse:
    if count < args.freq_threshold:
        to_be_removed.append(wordid)
low_freq_cnt = len(to_be_removed)
print(f'In terms of frequency, {low_freq_cnt} words are removed with threshold at {args.freq_threshold}')


if os.path.exists(os.path.join(args.final_data_path, 'token_story_occurence.pkl')):
    with open(os.path.join(args.final_data_path, 'token_story_occurence.pkl'), 'rb') as f:
        token_story_occurence = pickle.load(f)

for (wordid, cnt) in enumerate(token_story_occurence):
    if wordid not in to_be_removed and cnt <= args.occurrence_threshold:
        to_be_removed.append(wordid)
print(f'In terms of occurrences in stories, {len(to_be_removed) - low_freq_cnt} words are removed with threshold at {args.occurrence_threshold}')



# building the positive and negative data, shuffled
# uid_text_ids = uid_text_ids[:100]
random.shuffle(uid_text_ids)
uid_input_vector_list = []
for i in range( len( uid_text_ids) ): # loop over each challenge
    uid_text_tuple = uid_text_ids[i]
    if len(uid_text_tuple) ==2:
        uid, sent = uid_text_tuple

        if args.freq_filter_train:
            vectors = [ embedding_matrix_np[word_id, :] for word_id in sent if word_id not in to_be_removed]
        else:
            vectors = [ embedding_matrix_np[word_id, :] for word_id in sent]

        if len(vectors) > 0:
            vector_mean = np.mean( vectors, axis=0)
            uid_input_vector_list.append( (uid,vector_mean) )

print(f'Computed {len(uid_input_vector_list)} positive examples')


uid_input_vector_list_neg = []
indices = list(range(len(uid_input_vector_list)))
num_neg_samples = args.num_negative_samples
for idx in range(len(uid_input_vector_list)):
    # indices_candidate = [ i for i in indices if i != idx ]
    indices_candidate = indices
    neg_indices = random.sample( indices_candidate, num_neg_samples )
    neg_samples = [uid_input_vector_list[neg_i][1] for neg_i in neg_indices]
    neg_vector = np.mean(neg_samples, axis=0)
    uid_input_vector_list_neg.append(neg_vector)

print(f'Computed {len(uid_input_vector_list_neg)} negative examples')



TRAINING_MODEL = True
if TRAINING_MODEL:


    # training loop
    # normalize glove embeddings
    if args.use_norm_we:
        pass

    # set up hyperparameters
    net_params = {}
    # net_params['glove'] = norm_We
    net_params['mode'] = 'glove'
    net_params['embedding'] = embedding_matrix_np
    net_params['d_hid'] = args.d_hidden
    net_params['num_rows'] = args.num_topics # number of topics
    net_params['word_dropout_prob'] = args.dropout_rate
    net_params['vrev'] = id2word_dict # idx to word map
    net_params['device'] = device
    net_params['pred_world'] = False
    net_params['num_world'] = args.num_world


    net = DictionaryAutoencoder(net_params=net_params)
    net.to(device)


    # training specs
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    ortho_weight = args.ortho_weight
    world_clas_weight = args.world_clas_weight
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    interpret_interval = int(np.ceil(num_epochs / 10))


    # iterating through batches
    batch_intervals = [(start, start + batch_size) for start in range(0, len(uid_input_vector_list), batch_size)]
    # batch_intervals = batch_intervals[:100]
    split = int(np.ceil(len(batch_intervals) * 0.9))
    batch_intervals_train = batch_intervals[:split]
    batch_intervals_valid = batch_intervals[split:]

    print('\n' + '=' * 70)
    for epoch in range(num_epochs):


        # training
        net.train()
        train_mode = True
        print(f'Epoch {epoch}')
        run_epoch(net, optim, batch_intervals_train, uid_input_vector_list, uid_input_vector_list_neg, args, train_mode)


        # validation
        net.eval()
        train_mode = False
        with torch.no_grad():
            run_epoch(net, optim, batch_intervals_valid, uid_input_vector_list, uid_input_vector_list_neg, args, train_mode)



        if (epoch) % interpret_interval == 0:
            print('Topics with probability argmax: ')
            net.rank_vocab_for_topics(word_embedding_matrix=embedding_matrix_np, to_be_removed=to_be_removed)
            print('=' * 70)
        print()
        print()
        print()
        print('=' * 70)

    print('Finally after training')
    net.eval()
    print('Topics with probability argmax: ')
    net.rank_vocab_for_topics(word_embedding_matrix=embedding_matrix_np, to_be_removed=to_be_removed)
    print('=' * 70)


    if args.saved_model_path != '':
        with open(os.path.join(args.saved_model_path), 'wb') as f:
            torch.save(net, f)
            print(f'Saved model at { os.path.join(args.saved_model_path) }')


print('Done')



