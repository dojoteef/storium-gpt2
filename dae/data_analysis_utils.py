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
import tqdm
# spacy.load('en')
# from spacy.lang.en import English
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
add_words = ('failure_stakes', 'success_stakes', 'obstacle', 'challenge', 'outcome', 'something')
en_stop = en_stop.union(add_words)
import time
import langid
nlp = spacy.load("en_core_web_sm")
import concurrent.futures

# parser = English()







def tokenize_idtext(idtext):

    # id_tokens_list = []

    # t0 = time.time()
    # preprocessing data as in LDA
    # for idx, (uid, line_meta_tuple) in enumerate(idtext_list):
    uid, line_meta_tuple = idtext
    line = line_meta_tuple[0]
    if not line:
        return ()
    result, score = langid.classify(line)
    if result == 'en':
        tokens = prepare_text_for_lda(line) # takes long
            # tokens_strip_unk = [token for token in tokens if token in vocab_list]
            # id_tokens_list.append( (uid, tokens) )
        return (uid, tokens)
    else:
        return ()
        # if idx % interpret_interval == 0:
        #     t_taken = time.time() - t0
        #     print(f'[tokenize documents] {idx} / {num_doc} processed in time {t_taken}')
        #     t0 = time.time()


    # return id_tokens_list







def tokenize_spacy_fewer_pipeline(text):
    doc = nlp(text,  disable=["tagger", "parser"])
    lda_tokens = []
    ents = [e.text for e in doc.ents]
    if True:
        for token in doc:
            if token.text in ents:
                continue
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            else:
                lda_tokens.append(token.lower_)
    return lda_tokens



def tokenize(text):
    doc = nlp(text)
    lda_tokens = []
    ents = [e.text for e in doc.ents]
    if True:
        for token in doc:
            if token.text in ents:
                continue
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

# tokenize_spacy_fewer_pipeline
def prepare_text_for_lda(text):
    tokens = tokenize_spacy_fewer_pipeline(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens



def format_an_entry(entry, dict, world):
    entry_text = entry['description']

    result_tuple = (
        entry_text,
        {'game_pid': dict['game_pid'],
         'role': dict['game_pid'] + "_" + entry['role'],
         'type': 'entry',
         'world': world
         }

    )

    return result_tuple

def extract_content_from_file(file_name, full_export_data_path, type_text, world):
    '''

    :param file_name:
    :param full_export_data_path:
    :param type_text:
    :return: text_list is a list of tuples, each tuple consist of 2 things: (content of text, dict of metadata)
    '''
    file_name = os.path.join(full_export_data_path, file_name)
    text_list = []

    with open(file_name) as jsf:
        dict = json.load(jsf)
        scenes = dict['scenes']

        for sce_idx, scene in enumerate(scenes):  # each dict (a story) may contain several scenes
            entries = scene['entries']

            for entry_idx, entry in enumerate(entries):

                if type_text == 'entry' or type_text == 'entry_challenge_pooled':
                    result_tuple = format_an_entry(entry, dict, world)
                    text_list.append(result_tuple)

                if type_text == 'challenge' or type_text == 'entry_challenge_pooled':
                    # the challenge
                    try:
                        target_challenge_card = entry['target_challenge_card']
                        challenge_card_name = target_challenge_card['name']
                        challenge_card_description = target_challenge_card['description']
                        challenge_success = target_challenge_card['success_stakes']
                        challenge_failure = target_challenge_card['failure_stakes']
                        challenge_item = ' '.join([challenge_card_name + '.', challenge_card_description,
                                                   challenge_success, challenge_failure])
                    except Exception as e:
                        pass
                    else:

                        result_tuple = (
                            challenge_item,
                            {'game_pid': dict['game_pid'],
                             'role': dict['game_pid'] + "_" + entry['role'],
                             'type': 'challenge',
                             'world': world
                             }

                        )
                        text_list.append( result_tuple )
    return text_list



def read_from_json_export_and_organize_by_world(full_export_data_path, world_storyfile_dict, type_text):
    result_dict = {}
    for idx, (world, filenames) in enumerate(world_storyfile_dict.items()):
        result_dict[world] = {}
        for f_name in filenames:
            result_dict[world][f_name] = extract_content_from_file(f_name, full_export_data_path, type_text, world)
    return result_dict



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




def extract_textlist_from_world_story_dict(text_by_story_world_dict):
    '''
    :param text_by_story_world_dict:
    :return: uid_text_list, a list of tuple, each tuple is (idx, text_tuple), text tuple is defined as in
    '''
    all_text_list = []
    for idx, (world_name, stories) in enumerate(text_by_story_world_dict.items()):
        for story_fname, text_list in stories.items():
            all_text_list.extend(text_list)
    # all_text_list = list(set(all_text_list))
    uid_text_list = [ (idx, text_tuple) for idx, text_tuple in enumerate(all_text_list) ]
    return uid_text_list


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

            if idx % 22000 == 0:
                print(f'{idx} out of 22000000 done')
                print(f'{progress_cnt} / 100 done')
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


def _convert_token_2_ids(uid_token_tuple, word2id_dict, vocab_list):
    if len(uid_token_tuple) < 2:
        return ()
    uid,line = uid_token_tuple
    ids = [word2id_dict[token] for token in line if token in vocab_list] # considering oovs
    # ids = [[token] for token in line] # considering oovs
    return (uid, ids)



def convert_token_2_ids(word2id_dict, uid_tokens_list):

    # n = len(uid_tokens_list)
    # interval = np.ceil(n/50)

    # uid_text_id_list = []
    vocab_list = list(word2id_dict.keys())

    from itertools import repeat

    start = time.time()
    uid_text_ids= []
    chunk_size = 100
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_convert_token_2_ids, uid_tokens_list, repeat(word2id_dict), repeat(vocab_list), chunksize=chunk_size)
        for result in results:
            uid_text_ids.append(result)

    end = time.time()
    print(f'Chunksize = {chunk_size}, The time spent on conversion from text to ids on {len(uid_tokens_list)} examples '
          f'is {end - start}, ave {(end - start)/len(uid_tokens_list)} per example')

    return uid_text_ids




def text_to_topic(input_vector_list, model, device, batch_size=400):


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
