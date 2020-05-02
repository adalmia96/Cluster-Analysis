# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def combine_split_children():
    files = []
    index = 0
    with open('data/CBTest/data/cbt_train.txt', encoding='utf-8') as fp:
        data = fp.readlines()
    with open('data/CBTest/data/cbt_valid.txt', encoding='utf-8') as fp:
        data2 = fp.readlines()
    with open('data/CBTest/data/cbt_test.txt', encoding='utf-8') as fp:
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
             #print(sentence)

        if index % 20 == 0:
            files.append(sentence)
        else:
            files[int(index/20)] += sentence

        index += 1
    files = np.array(files)


    kf = KFold(n_splits=10, shuffle=True, random_state = 0)
    indices = list(kf.split(files))[0]

    train_valid = files[indices[0]]
    test = files[indices[1]]

    kf = KFold(n_splits=9, shuffle=True, random_state = 0)
    indices = list(kf.split(train_valid))[0]

    train = train_valid[indices[0]]
    valid = train_valid[indices[1]]

    return train, valid, test

def create_vocab_and_files_20news(stopwords, type, process=False):
    word_to_file = {}
    word_to_file_mult = {}

    train_data = fetch_20newsgroups(data_home='./data/', subset=type, remove=('headers', 'footers', 'quotes'))
    files = train_data['data'];
    #doc_to_word = np.zeros
    return create_vocab(stopwords, files, process)

def create_vocab(stopwords, data, process=False):

    word_to_file = {}
    word_to_file_mult = {}
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    process_files = []
    for file_num in range(0, len(data)):
        words = data[file_num].lower().split()
        #words = [w.strip() for w in words]
        proc_file = []

        for word in words:
            if "@" in word and "." in word:
                continue
            word = word.translate(strip_punct)
            word = word.translate(strip_digit)

            # check whether this is important
            word = word.strip()


            if word in stopwords:
                continue

            proc_file.append(word.strip())

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
        if len(word_to_file[word]) <= 5  or len(word) <= 3:
            word_to_file.pop(word, None)
            word_to_file_mult.pop(word, None)
    print("Files:" + str(len(data)))
    print("Vocab: " + str(len(word_to_file)))

    if process:
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


def get_tfidf_score(data, train_vocab, b_word):
    tf_idf_score = {}

    tfidf_vectorizer=TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(data)

    words = tfidf_vectorizer.get_feature_names()
    total_tf_idf = tfidf_vectorizer_vectors.toarray().sum(axis=0)

    vocab = set(words) & set(train_vocab.keys())
    vocab = set(b_word.keys()) & vocab

    for i, word in enumerate(words):
        if word in vocab:
            tf_idf_score[word] = total_tf_idf[i]

    return tf_idf_score

def topicwords_from_lda(fn):
    with open(fn, 'r') as f:
        topicwords = f.readlines()
    topicwords = [words.strip().split('\t')[2] for words in topicwords]
    topicwords = [w.split() for w in topicwords]
    return topicwords

# def create_vocab_and_files_children(stopwords, type):
#
#     word_to_file = {}
#     word_to_file_mult = {}
#     index = 0
#     strip_punct = str.maketrans("", "", string.punctuation)
#     strip_digit = str.maketrans("", "", string.digits)
#
#     for line in open('data/CBTest/data/cbt_'+type+'.txt', 'r'):
#         words = line.strip()
#         if "BOOK_TITLE" in words:
#             continue
#         elif  "CHAPTER" in words:
#             words = words.lower().split()[2:]
#         else:
#             words = words.lower().split()
#
#
#         for word in words:
#             word = word.translate(strip_punct)
#             word = word.translate(strip_digit)
#
#             if word in stopwords:
#                 continue
#             if word in word_to_file:
#                 word_to_file[word].add(int(index/20))
#                 word_to_file_mult[word].append(int(index/20))
#             else:
#                 word_to_file[word]= set()
#                 word_to_file_mult[word] = []
#
#                 word_to_file[word].add(int(index/20))
#                 word_to_file_mult[word].append(int(index/20))
#         index+=1
#
#     for word in list(word_to_file):
#         if len(word_to_file[word]) < 5 or len(word) <= 3:
#             word_to_file.pop(word, None)
#             word_to_file_mult.pop(word, None)
#     print(f"Length of {type} files:", int(index/20))
#     print("Vocab: " + str(len(word_to_file)))
#
#     return word_to_file, word_to_file_mult, index
