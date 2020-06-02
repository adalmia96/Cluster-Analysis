# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold
from nltk.corpus import reuters
import string
import numpy as np

import os
DATADIR="path-to/Cluster-Analysis/data"

def create_global_vocab(vocab_files):
    vocab_list = set(line.split()[0] for line in open(vocab_files[0]))
    for vocab in vocab_files:
        vocab_list = vocab_list & set(line.split()[0] for line in open(vocab))
    return vocab_list


def combine_split_children(type):
    files = []
    index = 0
    with open(os.path.join(DATADIR, 'CBTest/data/cbt_train.txt'), encoding='utf-8') as fp:
        data = fp.readlines()
    with open(os.path.join(DATADIR, 'CBTest/data/cbt_valid.txt'), encoding='utf-8') as fp:
        data2 = fp.readlines()
    with open(os.path.join(DATADIR, 'CBTest/data/cbt_test.txt'), encoding='utf-8') as fp:
        data3 = fp.readlines()
    data += "\n"
    data += data2
    data += "\n"
    data += data3

    for line in data:
        words = line.strip()
        if "BOOK_TITLE" in words:
            continue
        elif  "CHAPTER" in words:
            words = words.split()[2:]
        else:
            words = words.split()

        if "-RRB-" in words:
            words.remove("-RRB-")
        if "-LRB-" in words:
            words.remove("-LRB-")

        sentence = (" ".join(words) + "\n")
        if "-RCB-" in words:
             sentence = sentence[0:sentence.find("-")] + sentence[sentence.rfind("-")+1:]

        if index % 20 == 0:
            files.append(sentence)
        else:
            files[int(index/20)] += sentence

        index += 1
    files = np.array(files)


    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    indices = list(kf.split(files))[0]

    train_valid = files[indices[0]]
    test = files[indices[1]]

    kf = KFold(n_splits=4, shuffle=True, random_state = 0)
    indices = list(kf.split(train_valid))[0]

    train = train_valid[indices[0]]
    valid = train_valid[indices[1]]
    if type == "train":
        return train
    elif type == "valid":
        return valid
    else:
        return test

def create_files_20news(type):
    if type == "valid":
        type = "test"
    data = fetch_20newsgroups(data_home='./data/', subset=type, remove=('headers', 'footers', 'quotes'))
    files = data['data'];
    return files

def create_files_reuters(type):
    t = type
    if type == "valid":
        t = "test"

    documents = reuters.fileids()
    id = [d for d in documents if d.startswith(t)]
    files = np.array([reuters.raw(doc_id) for doc_id in id])

   # if type != "test":
   #     kf = KFold(n_splits=5, shuffle=True, random_state = 0)
   #     indices = list(kf.split(files))[0]
   #     train = files[indices[0]]
   #     valid = files[indices[1]]

   #     if type == "train":
   #         return train
   #     elif type == "valid":
   #         return valid
    return files

def create_files_children(type):
    files = combine_split_children(type)
    return files


def create_vocab_preprocess(stopwords, data, vocab, preprocess, process_data=False):
    word_to_file = {}
    word_to_file_mult = {}
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    process_files = []
    for file_num in range(0, len(data)):
        words = data[file_num].lower().translate(strip_punct).translate(strip_digit)
        words = words.split()
        #words = [w.strip() for w in words]
        proc_file = []

        for word in words:
            if word in stopwords or (word not in vocab and len(vocab)) or word =="dlrs" or word == "revs":
                continue
            if word in word_to_file:
                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)
            else:
                word_to_file[word]= set()
                word_to_file_mult[word] = []

                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)

        process_files.append(proc_file)

    for word in list(word_to_file):
        if len(word_to_file[word]) <= preprocess  or len(word) <= 3:
            word_to_file.pop(word, None)
            word_to_file_mult.pop(word, None)

    print("Files:" + str(len(data)))
    print("Vocab: " + str(len(word_to_file)))

    if process_data:
        vocab = word_to_file.keys()
        files = []
        for proc_file in process_files:
            fil = []
            for w in proc_file:
                if w in vocab:
                    fil.append(w)
            files.append(" ".join(fil))

        data = files

    return word_to_file, word_to_file_mult, data



def create_vocab_and_files(stopwords, dataset, preprocess, type, vocab):
    data = None
    if dataset == "20NG":
        data = create_files_20news(type)
    elif dataset == "children":
        data = create_files_children(type)
    elif dataset == "reuters":
        data = create_files_reuters(type)

    return create_vocab_preprocess(stopwords, data, vocab, preprocess)
