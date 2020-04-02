#!/usr/bin/python3
# Author: Suzanna Sia

# Standard imports
import random
import numpy as np
import pdb
import math
import os, sys
from sklearn.datasets import fetch_20newsgroups
import nltk.data
import string
# argparser
import argparse
#from distutils.util import str2bool
argparser = argparse.ArgumentParser()
argparser.add_argument('--layer', type=int)
args = argparser.parse_args()

# Custom imports
import time
import torch
from pytorch_transformers import *

def init():

    model, tokenizer = load_bert_models()
    input_ids = torch.tensor([tokenizer.encode('Here is some text to encode')])
    last_hidden_states = model(input_ids)[0][0]

    train_data = fetch_20newsgroups(data_home="./data", subset="train", remove=('headers',
        'footers', 'quotes'))

    files = train_data['data']

    with open('stopwords_en.txt', 'r', encoding="utf-8") as f:
        stopwords = f.readlines()

    stopwords = [s.strip() for s in stopwords]

    w2vb = {}
    w2vc = {}
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    vocab_counts = {}
    for i, fil in enumerate(files):
        fil = fil.translate(strip_punct).translate(strip_digit)
        words = [w.strip() for w in fil.split()]
        for word in words:
            if word in stopwords:
                continue

            if word in vocab_counts:
                vocab_counts[word].add(i)
            else:
                vocab_counts[word] = set()
                vocab_counts[word].add(i)

    valid_vocab = []
    for word in vocab_counts:
        if len(vocab_counts[word])>5 and len(word)>2:
            valid_vocab.append(word)

    start = time.time()
    print("vocab size:", len(valid_vocab))

    with torch.no_grad():
        for i, fil in enumerate(files):
            if i%(int(len(files)/100))==0:
                timetaken = np.round(time.time() - start, 1)
                print(f"{i}/{len(files)} done, elapsed(s): {timetaken}")
                sys.stdout.flush()

            sents = sent_detector.tokenize(fil)
            for sent in sents:
                words = tokenizer.tokenize(sent)
                if len(words)>50:
                    continue

                input_ids = torch.tensor([tokenizer.encode(sent)])
                embeds = model(input_ids)[-2:][1][args.layer][0]

                compound_word = []

                for w, word in enumerate(words):
                    if word.startswith("##"):
                        compound_word.append(word.strip('##'))

                    else:
                        if len(compound_word)==0:
                            pass
                        else:
                            word = "".join(compound_word)
                            compound_word = []

                    if word not in valid_vocab:
                        continue
                    
                    if word in w2vb:
                        w2vb[word] += embeds[w]
                        w2vc[word] += 1
                    else:
                        w2vb[word] = embeds[w]
                        w2vc[word] = 1

        eb_dump(i, w2vb, w2vc)
        w2vb = {}
        w2vc = {}

def eb_dump(i, w2vb, w2vc):
    all_vecs = []
    for word in w2vb:
        mean_vector = np.around(w2vb[word].detach().numpy()/w2vc[word], 5)
        #mean_vector = torch.mean(torch.stack(w2vb[word]), 0).detach().numpy()
        vect = np.append(word, mean_vector)

        if len(all_vecs)==0:
            all_vecs = vect
        else:
            all_vecs = np.vstack((all_vecs, vect))

    np.savetxt(f'embeds/bert_embeddings{i}-layer{args.layer}.txt', all_vecs, fmt = '%s', delimiter=" ")
    print(len(all_vecs))
    sys.stdout.flush()
    


def load_bert_models():
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)

    return model, tokenizer




if __name__ == "__main__":
    init()

