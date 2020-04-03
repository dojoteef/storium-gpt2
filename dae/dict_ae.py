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
from data_analysis_utils import (prepare_text_for_lda, read_from_json_export, tokenize_doc_list, build_reduced_glove_dict,
convert_token_2_ids, organize_story_by_world, read_from_json_export_and_organize_by_world, combine_two_text_by_story_world_dict)
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



parser.add_argument('--triplet_loss_margin', type=float, default=1.0)

parser.add_argument('--ortho_weight', type=float, default=1e-4)
parser.add_argument('--world_clas_weight', type=float, default=0.0)
parser.add_argument('--triplet_loss_weight', type=float, default=1.0)
parser.add_argument('--num_topics', type=int, default=20)
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--num_negative_samples', type=int, default=5)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--num_world', type=int, default=59) # 59 worlds, including the none
parser.add_argument('--ignore_none_world', action='store_true')

parser.add_argument('--bert_data_path', type=str, default='./bert_data')
parser.add_argument('--glove_data_path', type=str, default='/scratch/datasets/cc_glove')
parser.add_argument('--glove_file_name', type=str, default='glove.840B.300d.txt')
parser.add_argument('--splits_info_path', type=str, default='/scratch/datasets/storium')
parser.add_argument('--full_export_data_path', type=str, default='/scratch/datasets/storium/full_export_data')
parser.add_argument('--saved_model_path', type=str, default='')



parser.add_argument('--stored_data_path', type=str, default='./pooled_entry_challenge_data')
parser.add_argument('--type_text', type=str, default='entry_challenge_pooled')





args = parser.parse_args()
if args.type_text == 'entry':
    args.type_text = args.type_text + '_'





# fix random seed for python, numpy and torch to ensure reproducibility
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)



device = torch.device("cuda:" + str(args.device_id))
args.device = device
print('\n\n')
print('The hyperparameters: ')
print(args)



glove_data_path = args.glove_data_path
glove_file_name = args.glove_file_name  # the largest GloVe trained from common crawl (used the Wikipedia GloVe which does not work)
stored_data_path = args.stored_data_path
if not os.path.exists(stored_data_path):
    os.mkdir(stored_data_path)
splits_info_path = args.splits_info_path
full_export_data_path = args.full_export_data_path # the full storium story jsons



if args.type_text == 'entry_challenge_pooled':
    with open('./stored_data/text_by_story_world_dict.pkl', 'rb') as f:
        text_by_story_world_dict_challenge = pickle.load(f)

    with open('./entry_data/text_by_story_world_dict.pkl', 'rb') as f:
        text_by_story_world_dict_entry = pickle.load(f)

    text_by_story_world_dict = combine_two_text_by_story_world_dict(text_by_story_world_dict_challenge, text_by_story_world_dict_entry)
    with open(os.path.join(args.stored_data_path, 'text_by_story_world_dict.pkl'), 'wb') as filehandle:
        pickle.dump(text_by_story_world_dict, filehandle)


# Step 1 read in the splits info (training/validation/test files path) and extract challenges from all sources
print('\n\n')
text_by_story_world_fname = os.path.join( args.stored_data_path, f'text_by_story_world_dict.pkl' )
if os.path.exists( text_by_story_world_fname ):
    with open( text_by_story_world_fname, 'rb') as f:
        text_by_story_world_dict = pickle.load(f)
    print(f'{text_by_story_world_fname} with {len(text_by_story_world_dict)} worlds are loaded from pre-processed')
else:
    with open( os.path.join('./text_data', 'world_storyfile_dict.pkl'), 'rb') as f:
        world_storyfile_dict = pickle.load(f)
    text_by_story_world_dict = read_from_json_export_and_organize_by_world(full_export_data_path, world_storyfile_dict, args.type_text)
    # challenge_doc_list = read_from_json_export(splits_info_path, full_export_data_path)
    with open(os.path.join(args.stored_data_path, 'text_by_story_world_dict.pkl'), 'wb') as filehandle:
        pickle.dump(text_by_story_world_dict, filehandle)
    print(f'{text_by_story_world_fname} with {len(text_by_story_world_dict)} text docs are read from json and saved')





# args.pred_world_label = True
# Step 1.5 if we are to predict_world label
if True:
    if not os.path.exists(os.path.join(args.stored_data_path, 'text_world_tuples_list.pkl')):
        with open(os.path.join(args.stored_data_path, 'text_by_story_world_dict.pkl'), 'rb') as f:
            text_by_story_world_dict = pickle.load(f)

        text_world_tuples_list = []
        for idx, (world_name, stories) in enumerate( text_by_story_world_dict.items() ):

            world_text_list =[]
            for story_fname, text_list in stories.items():
                if args.type_text != 'challenge':
                    if len(text_list) > 0:

                        entry_list = []
                        for item in text_list:
                            if isinstance(item, tuple):
                                entry_list.append(item[0])
                            if isinstance(item, str):
                                entry_list.append(item)
                        world_text_list.extend(entry_list)
                else:
                    world_text_list.extend(text_list)
            world_text_list = list(set(world_text_list))

            for sent in world_text_list:
                try:
                    # result = detect(sent)
                    # if result == 'en':
                    if True:
                        text_world_tuples_list.append( (sent, world_name) )
                except:
                    pass
            print(f'world {idx}: {world_name} done')

        # text_world_dict = dict(text_world_tuples_list)
        text_world_tuples_list = [ (sent, world_name) for (sent, world_name) in text_world_tuples_list if sent]
        text_doc_list = [sent for (sent, world_name) in text_world_tuples_list]
        world_label_ordered_by_doc = [world_name for (sent, world_name) in text_world_tuples_list]
        world_counter = Counter(world_label_ordered_by_doc)
        world_counter_ranked = world_counter.most_common()
        for (world, num_challenges) in world_counter_ranked:
            print(f'World {world} has number of challenges = {num_challenges}')

        with open(os.path.join(args.stored_data_path, 'text_world_tuples_list.pkl'), 'wb') as f:
            pickle.dump(text_world_tuples_list, f)
    else:
        with open(os.path.join(args.stored_data_path, 'text_world_tuples_list.pkl'), 'rb') as f:
            text_world_tuples_list = pickle.load(f)
        text_doc_list = [sent for (sent, world_name) in text_world_tuples_list]
        world_label_ordered_by_doc = [world_name for (sent, world_name) in text_world_tuples_list]
    print('Loaded story and world info')





# Step 2 tokenize, stem and lemmatize the docs
print('\n\n')
if os.path.exists( os.path.join(stored_data_path, 'text_tokens_list.pkl') ):
    with open(os.path.join(stored_data_path, 'text_tokens_list.pkl'), 'rb') as filehandle:
        text_tokens_list = pickle.load(filehandle)
    print(f'{len(text_tokens_list)} text tokens (tokenized) are loaded from pre-processed')
else:
    text_tokens_list = tokenize_doc_list(text_doc_list)
    with open(os.path.join(stored_data_path, 'text_tokens_list.pkl'), 'wb') as filehandle:
        pickle.dump(text_tokens_list, filehandle)
    print(f'{len(text_tokens_list)} docs are tokenized and lemmatized')


# Step 2.5 get unique vocabuaries from tokens_list
print('\n\n')
from data_analysis_utils import flatten
if os.path.exists(os.path.join(stored_data_path, 'uniq_vocab_list.pkl')):
    uniq_vocab_list = pickle.load( open( os.path.join(stored_data_path, 'uniq_vocab_list.pkl'), 'rb'))
    print(f'From the documents, there are {len(uniq_vocab_list)} unique vocabs (loaded from preprocessed)')

else:
    uniq_vocab_list = list(set(list(flatten(text_tokens_list))))
    pickle.dump(uniq_vocab_list, open( os.path.join(stored_data_path, 'uniq_vocab_list.pkl') , 'wb'))
    print(f'From the documents, there are {len(uniq_vocab_list)} unique vocabs (obtained by using set)')



# Step 3 build 'reduced'(limiting to vocab present in dataset) glove based dictionary
print('\n\n')
if os.path.exists( os.path.join(stored_data_path, 'embedding_matrix.npy') ) and \
    os.path.exists( os.path.join(stored_data_path, 'id2word_dict.pkl') ) and \
    os.path.exists( os.path.join(stored_data_path, 'word2id_dict.pkl') ):
    with open(os.path.join(stored_data_path, 'word2id_dict.pkl'), 'rb') as f:
        word2id_dict = pickle.load(f)

    with open(os.path.join(stored_data_path, 'id2word_dict.pkl'), 'rb') as f:
        id2word_dict = pickle.load(f)

    embedding_matrix_np = np.load(os.path.join(stored_data_path, 'embedding_matrix.npy'))

    print('Loaded word2id_dict with length: ')
    print(len(word2id_dict))
    print('Loaded id2word_dict with length: ')
    print(len(id2word_dict))
    print('Loaded pre-trained embedding with shape: ')
    print(embedding_matrix_np.shape)
else:
    print('Building embedding matrix and dictionaries (this may take 30 minutes)')
    build_reduced_glove_dict(uniq_vocab_list, glove_data_path, glove_file_name, stored_data_path)

    with open(os.path.join(stored_data_path, 'word2id_dict.pkl'), 'rb') as f:
        word2id_dict = pickle.load(f)

    with open(os.path.join(stored_data_path, 'id2word_dict.pkl'), 'rb') as f:
        id2word_dict = pickle.load(f)

    embedding_matrix_np = np.load(os.path.join(stored_data_path, 'embedding_matrix.npy'))

    print('Loaded word2id_dict with length: ')
    print(len(word2id_dict))
    print('Loaded id2word_dict with length: ')
    print(len(id2word_dict))
    print('Loaded pre-trained embedding with shape: ')
    print(embedding_matrix_np.shape)


# Step 4 word tokens to word IDs
if os.path.exists( os.path.join(stored_data_path, 'text_ids.npy') ):
    with open( os.path.join(stored_data_path, 'text_ids.npy'), 'rb') as f:
        text_ids = pickle.load(f)
    print(f'Load previously converted token ids with number of data instances = {len(text_ids)}')
else:
    text_ids = convert_token_2_ids(word2id_dict, text_tokens_list)
    print(f'Length of the text_ids list = {len(text_ids)}')
    with open( os.path.join(stored_data_path, 'text_ids.npy'), 'wb') as f:
        pickle.dump(text_ids, f)
    print(f'Save word ids converted from tokens with number of data points = {len(text_ids)}')



textID_world_list = [ (sent, world_name) for (sent, world_name) in zip(text_ids, world_label_ordered_by_doc) if len(sent)>0 ]


print(f'After removing empty list from text_ids, we have {len(textID_world_list)} data points')

# create a dictionary mapping from world label to an label id
all_labels = [world_name for (sent, world_name) in textID_world_list]
all_labels = list(set(all_labels))

world_label2id_dict = { all_labels[i]:i for i in range(len(all_labels)) }
id2world_label_dict = { v:k for k,v in world_label2id_dict.items() }
for (id, world_name) in id2world_label_dict.items():
    print(f'{id}: {world_name}')
none_id = world_label2id_dict['None']
args.none_id = none_id

if args.focus_on_world == '':
    textID_worldID_list = [ (sent, world_label2id_dict[world_name]) for (sent, world_name) in textID_world_list]
    print("Does not focus on any particular world")
else:
    focus_on_world_id = world_label2id_dict[args.focus_on_world]
    textID_worldID_list = [ (sent, world_label2id_dict[world_name])
                            for (sent, world_name) in textID_world_list
                            if world_name == args.focus_on_world]

    print(f'Focus on world {args.focus_on_world} with {len(textID_worldID_list)} texts')

random.shuffle(textID_worldID_list)

text_ids = [sent for (sent, _) in textID_world_list]

####################################################################################
# Add enforcing hard examples later!
# when a given example belongs to a big world, select other examples from that particular world
#
####################################################################################
input_vector_world_id_list = []
for i in range( len(textID_worldID_list) ): # loop over each challenge

    sent = textID_worldID_list[i][0]
    if len(sent) > 0:
        vectors = [ embedding_matrix_np[word_id, :] for word_id in sent ]
        vector_mean = np.mean(vectors, axis=0)
    input_vector_world_id_list.append((vector_mean, textID_worldID_list[i][1]))




input_vector_list_neg = []
indices = list(range(len(input_vector_world_id_list)))
num_neg_samples = args.num_negative_samples




for idx in range(len(input_vector_world_id_list)):
    # indices_candidate = [ i for i in indices if i != idx ]
    indices_candidate = indices
    neg_indices = random.sample( indices_candidate, num_neg_samples )
    neg_samples = [input_vector_world_id_list[neg_i][0] for neg_i in neg_indices]
    neg_vector = np.mean(neg_samples, axis=0)
    input_vector_list_neg.append(neg_vector)




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
    net_params['pred_world'] = True
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
    batch_intervals = [(start, start + batch_size) for start in range(0, len(input_vector_world_id_list), batch_size)]
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
        run_epoch(net, optim, batch_intervals_train, input_vector_world_id_list, input_vector_list_neg, args, train_mode)


        # validation
        net.eval()
        train_mode = False
        with torch.no_grad():
            run_epoch(net, optim, batch_intervals_train, input_vector_world_id_list, input_vector_list_neg, args, train_mode)



        if (epoch) % interpret_interval == 0:
            print('Topics with nearest neighbour decoding: ')
            net.interpret_dictionary()
            print('=' * 70)
            print('Topics with probability argmax: ')
            net.rank_vocab_for_topics(word_embedding_matrix=embedding_matrix_np)
            print('=' * 70)
        print()
        print()
        print()
        print('=' * 70)

    print('Finally after training')
    net.eval()
    print('Topics with nearest neighbour decoding: ')
    net.interpret_dictionary()
    print('=' * 70)
    print('Topics with probability argmax: ')
    net.rank_vocab_for_topics(word_embedding_matrix=embedding_matrix_np)


    if args.saved_model_path != '':
        with open(os.path.join(args.saved_model_path), 'wb') as f:
            torch.save(net, f)
            print(f'Saved model at { os.path.join(args.saved_model_path) }')


print('Done')



