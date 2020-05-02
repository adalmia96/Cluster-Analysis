from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold
from nltk.corpus import reuters
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def combine_split_children():
    files = []
    index = 0
    with open('data/CBTest/data/cbt_train.txt') as fp:
        data = fp.readlines()
    with open('data/CBTest/data/cbt_valid.txt') as fp:
        data2 = fp.readlines()
    with open('data/CBTest/data/cbt_test.txt') as fp:
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


    kf = KFold(n_splits=5, shuffle=True, random_state = 0)
    indices = list(kf.split(files))[0]

    train_valid = files[indices[0]]
    test = files[indices[1]]

    kf = KFold(n_splits=4, shuffle=True, random_state = 0)
    indices = list(kf.split(train_valid))[0]

    train = train_valid[indices[0]]
    valid = train_valid[indices[1]]

    return train, valid, test

def create_vocab_and_files_20news(stopwords, type):
    train_data = fetch_20newsgroups(data_home='./data/', subset=type, remove=('headers', 'footers', 'quotes'))
    files = train_data['data'];
    #doc_to_word = np.zeros
    return create_vocab(stopwords, files)


def create_vocab_and_files_reuters(stopwords, type):
    documents = reuters.fileids()
    id = [d for d in documents if d.startswith(type)]
    files = [reuters.raw(doc_id) for doc_id in id]
    return create_vocab(stopwords, files)


def create_vocab(stopwords, data):
    word_to_file = {}
    word_to_file_mult = {}
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    for file_num in range(0, len(data)):
        words = data[file_num].lower().split()
        #words = [w.strip() for w in words]
        for word in words:
            if "@" in word and "." in word:
                continue
            word = word.translate(strip_punct)
            word = word.translate(strip_digit)
            if word in stopwords:
                continue

            if word in word_to_file:
                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)
            else:
                word_to_file[word]= set()
                word_to_file_mult[word] = []

                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)

    for word in list(word_to_file):
        if len(word_to_file[word]) <= 5  or len(word) <= 3:
            word_to_file.pop(word, None)
            word_to_file_mult.pop(word, None)
    print("Files:" + str(len(data)))
    print("Vocab: " + str(len(word_to_file)))

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
