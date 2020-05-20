import string
import random
import numpy as np

from sklearn.datasets import fetch_20newsgroups

import preprocess

def init():
    with open('stopwords_en.txt', 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
    stopwords = [s.strip() for s in stopwords]
    strip_punct = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    strip_digit = str.maketrans("", "", string.digits)

    vocab = preprocess.create_global_vocab(["embeds/reuters-bert-layer12-average.full_vocab.fix"])#, "embeds/20NG-elmo.full_vocab.punc_respace"]) 

    word_to_file, word_to_file_mult, train = preprocess.create_vocab_and_files(stopwords, "reuters", 5, "train", vocab)

    for i, file in enumerate(train):
        fil = train[i].translate(strip_punct).translate(strip_digit)
        train[i] = ""
        words = [w.strip() for w in fil.split()]
        for word in words:
            if word in word_to_file:
                train[i] += (word  + " ")

    for i, file in enumerate(train):
        f = open("data/train/train"+str(i)+".txt","w")
        f.write(file)
        f.close()


if __name__ == "__main__":
    init()
