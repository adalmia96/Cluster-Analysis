import string
import random
import numpy as np

from sklearn.datasets import fetch_20newsgroups

import preprocess

def init():
    with open('stopwords_en.txt', 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
    stopwords = [s.strip() for s in stopwords]
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    #train, valid, test = preprocess.combine_split_children()

    train_data = fetch_20newsgroups(data_home='./data/', subset='train', remove=('headers', 'footers', 'quotes'))
    train = train_data['data'];

    word_to_file, word_to_file_mult, num = preprocess.create_vocab(stopwords, train)

    for i, file in enumerate(train):
        print(train[i])
        fil = train[i].translate(strip_punct).translate(strip_digit)
        train[i] = ""
        words = [w.strip() for w in fil.split()]
        for word in words:
            if word in word_to_file:
                train[i] += (word  + " ")
        print(train[i])
        break

    #for i, file in enumerate(train):
    #    f = open("train"+str(i)+".txt","w")
    #    f.write(file)
    #    f.close()


if __name__ == "__main__":
    init()
