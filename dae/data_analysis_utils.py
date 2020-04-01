'''

    A bunch of helpers for data analysis

'''


import os
import torch
import pickle
import json
import nltk
import spacy
import numpy as np
from collections import Iterable

spacy.load('en')
from spacy.lang.en import English
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
add_words = ('failure_stakes', 'success_stakes', 'obstacle', 'challenge', 'outcome', 'something')
en_stop = en_stop.union(add_words)

parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# lemmna via morphy
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

# lemmna via wordnetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens



def read_from_json_export(splits_info_path, full_export_data_path):

    mode_list = ['train', 'validation', 'test']
    filenames_list = []
    for mode in mode_list:
        with open(os.path.join(splits_info_path, f'{mode}_filenames.txt'), 'r') as f:
            filenames_list.extend(f.readlines())

    print('\n\n')
    print(f'Reading from scratch from: {len(filenames_list)} json files')

    challenge_doc_list = []
    for file_idx, file_name in enumerate(filenames_list):

        # if file_idx == 100:
        #     break

        if file_idx % 100 == 0:
            print(f'{file_idx} / {len(filenames_list)} processing ...')

        file_name = file_name.strip('\n')
        file_name = os.path.join( full_export_data_path, '/'.join(file_name.split('/')[1:]))

        with open(file_name) as jsf:
            dict = json.load(jsf)
            scenes = dict['scenes']

            for sce_idx, scene in enumerate(scenes):  # each dict (a story) may contain several scenes
                entries = scene['entries']

                for entry_idx, entry in enumerate(entries):

                    # the challenge
                    try:
                        target_challenge_card = entry['target_challenge_card']
                        challenge_name_space = target_challenge_card['namespace']
                        challenge_card_name = target_challenge_card['name']
                        challenge_card_description = target_challenge_card['description']
                        challenge_success = target_challenge_card['success_stakes']
                        challenge_failure = target_challenge_card['failure_stakes']
                        challenge_item = ' '.join([challenge_card_name + '.', challenge_card_description,
                                                  challenge_success, challenge_failure])
                    except Exception as e:
                        pass
                    else:
                        challenge_doc_list.append(challenge_item)

    return challenge_doc_list



def read_from_json_export_by_story(splits_info_path, full_export_data_path):

    mode_list = ['train', 'validation', 'test']
    filenames_list = []
    for mode in mode_list:
        with open(os.path.join(splits_info_path, f'{mode}_filenames.txt'), 'r') as f:
            filenames_list.extend(f.readlines())

    print('\n\n')
    print(f'Reading from scratch from: {len(filenames_list)} json files')

    challenge_doc_list_by_story = []
    for file_idx, file_name in enumerate(filenames_list):

        # if file_idx == 5:
        #     break

        if file_idx % 100 == 0:
            print(f'{file_idx} / {len(filenames_list)} processing ...')

        file_name = file_name.strip('\n')
        file_name = os.path.join( full_export_data_path, '/'.join(file_name.split('/')[1:]))

        with open(file_name) as jsf:

            challenge_doc_list = ['_BOS_']

            dict = json.load(jsf)
            scenes = dict['scenes']

            for sce_idx, scene in enumerate(scenes):  # each dict (a story) may contain several scenes
                entries = scene['entries']

                for entry_idx, entry in enumerate(entries):

                    # the challenge
                    try:
                        target_challenge_card = entry['target_challenge_card']
                        challenge_name_space = target_challenge_card['namespace']
                        challenge_card_name = target_challenge_card['name']
                        challenge_card_description = target_challenge_card['description']
                        challenge_success = target_challenge_card['success_stakes']
                        challenge_failure = target_challenge_card['failure_stakes']
                        challenge_item = ' '.join([challenge_name_space + '.', challenge_card_name + '.', challenge_card_description,
                                                  challenge_success, challenge_failure])
                    except Exception as e:
                        pass
                    else:
                        # print('here')
                        challenge_doc_list.append(challenge_item)

            challenge_doc_list.append('_EOS_')
        challenge_doc_list_by_story.append(challenge_doc_list)
    return challenge_doc_list_by_story


def format_an_entry(entry, game_pid):
    entry_text = entry['description']
    character = entry['role'] # narrator or character:1570
    return (entry_text, game_pid + '_' + character)


def extract_content_from_file(file_name, full_export_data_path, type_text):
    # file_name = file_name.strip('\n')
    file_name = os.path.join(full_export_data_path, file_name)

    text_list = []

    with open(file_name) as jsf:
        dict = json.load(jsf)
        scenes = dict['scenes']

        for sce_idx, scene in enumerate(scenes):  # each dict (a story) may contain several scenes
            entries = scene['entries']

            for entry_idx, entry in enumerate(entries):

                if type_text == 'entry_':
                    result = format_an_entry(entry, dict['game_pid'])
                    text_list.append(result)

                else:
                    # the challenge
                    try:
                        target_challenge_card = entry['target_challenge_card']
                        challenge_name_space = target_challenge_card['namespace']
                        challenge_card_name = target_challenge_card['name']
                        challenge_card_description = target_challenge_card['description']
                        challenge_success = target_challenge_card['success_stakes']
                        challenge_failure = target_challenge_card['failure_stakes']
                        challenge_item = ' '.join([challenge_card_name + '.', challenge_card_description,
                                                   challenge_success, challenge_failure])
                    except Exception as e:
                        pass
                    else:
                        text_list.append(challenge_item)

    return text_list


def extract_textlist_from_world_story_dict(text_by_story_world_dict, type_text):
    all_text_list = []
    for idx, (world_name, stories) in enumerate(text_by_story_world_dict.items()):

        world_text_list = []
        for story_fname, text_list in stories.items():
            if type_text == 'entry_':
                if len(text_list) > 0:
                    entry_list = [items[0] for items in text_list]
                    world_text_list.extend(entry_list)
            else:
                world_text_list.extend(text_list)
        world_text_list = list(set(world_text_list))
        all_text_list.extend(world_text_list)
    return list(set(all_text_list))




def organize_story_by_world(splits_info_path, full_export_data_path):
    mode_list = ['train', 'validation', 'test']
    filenames_list = []
    for mode in mode_list:
        with open(os.path.join(splits_info_path, f'{mode}_filenames.txt'), 'r') as f:
            filenames_list.extend(f.readlines())

    print('\n\n')
    print(f'Reading from scratch from: {len(filenames_list)} json files')


    world_storyfile_dict = {}

    for file_idx, file_name in enumerate(filenames_list):

        # if file_idx == 100:
        #     break

        if file_idx % 100 == 0:
            print(f'{file_idx} / {len(filenames_list)} processing ...')

        file_name = file_name.strip('\n')
        file_name_short = '/'.join(file_name.split('/')[1:])

        file_name = os.path.join(full_export_data_path, file_name_short)

        with open(file_name) as jsf:
            dict = json.load(jsf)
            world_name = str(dict['details']['world']['name'])
            if world_name not in world_storyfile_dict:
                world_storyfile_dict[world_name] = [file_name_short]
            else:
                world_storyfile_dict[world_name].append(file_name_short)
    return world_storyfile_dict


def read_from_json_export_and_organize_by_world(full_export_data_path, world_storyfile_dict, type_text):
    result_dict = {}
    for idx, (world, filenames) in enumerate(world_storyfile_dict.items()):
        result_dict[world] = {}
        for f_name in filenames:
            result_dict[world][f_name] = extract_content_from_file(f_name, full_export_data_path, type_text)

    return result_dict



def tokenize_doc_list(doc_list):
    num_doc = len(doc_list)
    interpret_interval = int(np.ceil(num_doc / 10))

    challenge_text_list = []

    # preprocessing data as in LDA
    for idx, line in enumerate(doc_list):
        tokens = prepare_text_for_lda(line) # takes long
        # tokens_strip_unk = [token for token in tokens if token in vocab_list]
        challenge_text_list.append(tokens)
        # if idx % num_doc == 0:
            # print(f'[tokenize documents] {idx} / {num_doc} processed')

    return challenge_text_list


def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:
             yield item


def build_reduced_glove_dict(uniq_vocab, glove_data_path, glove_file_name, stored_data_path):
    challenge_text_list_uniq = uniq_vocab

    # print('Building reduced glove word embedding matrix according to uniq vocabs present in the dataset')
    # print('Loading Glove from: ')
    # print(glove_data_path + '  ----   ' + glove_file_name)
    # embedding_dict = {}
    word2id_dict = {}
    id2word_dict = {}
    embedding_list = []

    # use downloaded glove to build dictionaries
    glove_word_list = []
    progress_cnt = 0
    with open(os.path.join(glove_data_path, glove_file_name), 'r') as f:
        cnt = 0
        for idx, line in enumerate(f):
            values = line.split()
            # print(values)
            word = ''.join(values[0:-300])
            # print(word)
            if word in challenge_text_list_uniq and word not in list(word2id_dict.keys()):
                coefs = np.asarray(values[-300:], dtype='float32')
                embedding_list.append(coefs)

                word2id_dict[word] = cnt
                id2word_dict[cnt] = word
                cnt += 1

            # if idx % 22000 == 0:
            #     print(f'{idx} out of 22000000 done')
            #     print(f'{progress_cnt} / 100 done')
                progress_cnt += 1

    embedding_matrix_np = np.array(embedding_list)


    unk_id = embedding_matrix_np.shape[0]
    embedding_matrix_np = np.vstack((embedding_matrix_np, np.mean(embedding_matrix_np, axis=0)))
    word2id_dict['[UNK]'] = unk_id # ('[UNK]', 27200)
    id2word_dict[unk_id] = '[UNK]'
    # print(f'Added {id2word_dict[unk_id]} token at index {unk_id}')


    pad_id = embedding_matrix_np.shape[0]
    embedding_matrix_np = np.vstack((embedding_matrix_np, np.zeros(shape=(embedding_matrix_np.shape[1],), dtype='float32')))
    word2id_dict['[PAD]'] = pad_id # ('[PAD]', 27201)
    id2word_dict[pad_id] = '[PAD]'
    print(f'Added {id2word_dict[pad_id]} token at index {pad_id}')

    print('After adding tokens: ')


    # save dictionaries to disk
    with open(os.path.join(stored_data_path, 'word2id_dict.pkl'), 'wb') as f:
        pickle.dump(word2id_dict, f)
    print(f'Saving word2id dict of {len(word2id_dict)}')

    with open(os.path.join(stored_data_path, 'id2word_dict.pkl'), 'wb') as f:
        pickle.dump(id2word_dict, f)
    print(f'Saving id2word dict of {len(id2word_dict)}')

    # print(embedding_matrix_np.shape)
    np.save(os.path.join(stored_data_path, 'embedding_matrix'), embedding_matrix_np)
    print(f'Saving embedding matrix of shape {embedding_matrix_np.shape[0]}-by-{embedding_matrix_np.shape[1]}')




def convert_token_2_ids(word2id_dict, challenge_tokens_list):
    # print()
    # print('Converting tokens list to word ID list ...')
    challenge_id_list = []

    num_doc = len(challenge_tokens_list)
    vocab_list = list(word2id_dict.keys())
    for idx, line in enumerate(challenge_tokens_list):
        ids = [word2id_dict[token] for token in line if token in vocab_list] # considering oovs
        challenge_id_list.append(ids)
        # if idx % 3000 == 0:
            # print(f'[convert token to id] {idx} / {num_doc} processed')

    return challenge_id_list




def text_to_topic(challenge_doc_list, word2id_dict, embedding_matrix_np, model, device, batch_size):
    # Step 2 tokenize, stem and lemmatize the docs (and also get the uniq vocab along the way)
    challenge_tokens_list = tokenize_doc_list(challenge_doc_list)
    # Step 4 word tokens to word IDs
    text_ids = convert_token_2_ids(word2id_dict, challenge_tokens_list)

    input_vector_list = []
    for i in range(len(text_ids)):  # loop over all challenges

        try:
            sent = text_ids[i]
            if len(sent) > 0:
                vectors = [embedding_matrix_np[word_id, :] for word_id in sent]
                vector_mean = np.mean(vectors, axis=0)
            input_vector_list.append(vector_mean)
        except UnboundLocalError as e:
            print('error caught here')
            print(sent)

    batch_intervals = [(start, start + batch_size) for start in range(0, len(input_vector_list), batch_size)]

    result = np.empty((0), dtype=int)
    for b_idx, (start, end) in enumerate(batch_intervals):
        batch_data = input_vector_list[start:end]
        batch_data_t = torch.FloatTensor(np.array(batch_data)).to(device)
        with torch.no_grad():
            scores = model.evaluate_topics(batch_data_t)
        _, ind = torch.max(scores, 1)
        ind_np = ind.data.cpu().numpy()
        result = np.concatenate((result, ind_np))
    return result


def get_text_given_world(world_story_dict):
    result = []
    for story_filename, text_list in world_story_dict.items():
        result.extend(text_list)
    return result


def compute_topic_transition_matrix(topic_id_list, num_topics):
    '''
    :param topic_list: a list of int
    :return: the transition matrix based on count, not probability

    '''

    # compute the transition matrix
    topic_transition_matrix = np.zeros(shape=(num_topics, num_topics))

    length_story = len(topic_id_list)

    for idx, topic_id in enumerate(topic_id_list):

        if idx == length_story - 1: # the last topic has no transition
            break

        curr = topic_id_list[idx]
        next = topic_id_list[idx + 1]
        topic_transition_matrix[curr][next] += 1
    return topic_transition_matrix


def filtere_off_unk_topics(topic_transition_matrix, idx_filter_off):
    N, _ = topic_transition_matrix.shape
    for idx in idx_filter_off:
        topic_transition_matrix[idx, :] = np.zeros((N,))
        topic_transition_matrix[:, idx] = np.zeros((N,))

    return topic_transition_matrix


def combine_two_text_by_story_world_dict(text_by_story_world_dict_1, text_by_story_world_dict_2):
    new_dict = {}
    keys_1 = list(text_by_story_world_dict_1.keys())
    keys_2 = list(text_by_story_world_dict_2.keys())
    keys_new = list(set(list(keys_1 + keys_2)))
    for world in keys_new:
        story_fnames_1 = list(text_by_story_world_dict_1[world].keys())
        story_fnames_2 = list(text_by_story_world_dict_2[world].keys())
        story_fnames_new = list(set(story_fnames_1 + story_fnames_2))
        world_dict ={}
        for story_fname in story_fnames_new:
            # world_dict[story_fname] = []
            pooled_list = []
            if story_fname in story_fnames_1:
                pooled_list += text_by_story_world_dict_1[world][story_fname]
            if story_fname in story_fnames_2:
                pooled_list += text_by_story_world_dict_2[world][story_fname]
            world_dict[story_fname] = pooled_list
        new_dict[world] = world_dict
    return new_dict
