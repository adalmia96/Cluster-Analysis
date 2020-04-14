#!/usr/bin/python3
# Author: Suzanna Sia

# Standard imports
import random
import numpy as np
#import pdb
import math
import os, sys

# argparser
import argparse
from sklearn.datasets import fetch_20newsgroups 
import nltk
import string
#argparser = argparser.ArgumentParser()
#argparser.add_argument('--x', type=float, default=0)

# Custom imports

class Pipeline():

    def __init__(self):
        self.vocab = []
        self.nfiles = 0
        self.word_to_file = {}
        self.stopwords = set()
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.data = None

    def load_stopwords(self, fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            self.stopwords = f.readlines()
        self.stopwords = [w.strip() for w in self.stopwords]
        #self.stopwords = set(line.strip() for line in open(fpath))

    def get_data(self):
        return self.data

    def create_vocab_to_files(self, datatype):
        strip_punct = str.maketrans("", "", string.punctuation)
        strip_digit = str.maketrans("", "", string.digits)

        data = fetch_20newsgroups(data_home="./data", subset=datatype, remove=('headers',
            'footers', 'quotes'))

        data = data['data']
        self.data = data

        for i, fil in enumerate(data):

            fil = fil.translate(strip_punct).translate(strip_digit)
            words = [w.strip() for w in fil.lower().split()]
            

            for word in words:
        #        word = word.translate(strip_punct)
        #        word = word.translate(strip_digit)

                if word in self.stopwords:
                    continue
                if word in self.word_to_file:
                    self.word_to_file[word].add(i)
                else:
                    self.word_to_file[word] = set()
                    self.word_to_file[word].add(i)

        valid_vocab = []
        for word in self.word_to_file:
            if len(self.word_to_file[word])>=5 and len(word)>2:
                valid_vocab.append(word)

        print("vocab size:", len(valid_vocab))
        return valid_vocab, len(data), self.word_to_file



