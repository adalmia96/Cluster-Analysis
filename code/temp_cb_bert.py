#!/usr/bin/python3

### Standard imports
#import random
#import numpy as np
#import pdb
#import math
#import os
#import sys
#import argparse

### Third Party imports

### Local/Custom imports

#from distutils.util import str2bool
#argparser = argparser.ArgumentParser()
#argparser.add_argument('--x', type=float, default=0)


import preprocess
stopwords = set(line.strip() for line in open('stopwords_en.txt'))

train, valid, test = preprocess.combine_split_children()
word_to_file, _, data = preprocess.create_vocab(stopwords, train)

valid_vocab = word_to_file.keys()
print("vocab size:", len(valid_vocab))

encoder = BertWordFromTextEncoder(valid_vocab=valid_vocab)
encoder.encode_docs(docs=data, save_fn="embeds/cb_bert_layer12_average")
