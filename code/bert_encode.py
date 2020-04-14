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
import preprocess
from pytorch_transformers import *

def init():

    #input_ids = torch.tensor([tokenizer.encode('Here is some text to encode')])
    #last_hidden_states = model(input_ids)[0][0]

    w2vb = {}
    w2vc = {}

    prep = preprocess.Pipeline()
    prep.load_stopwords('stopwords_en.txt')
    valid_vocab, len_train, train_word_to_file = prep.create_vocab_to_files('train')
#    pdb.set_trace()
    files = prep.get_data()
    model, tokenizer = load_bert_models()
    start = time.time()

    #print("vocab size:", len(valid_vocab))

    unique_vocab = set()
    with torch.no_grad():
        for i, fil in enumerate(files):
            if i%(int(len(files)/100))==0:
                timetaken = np.round(time.time() - start, 1)
                print(f"{i}/{len(files)} done, elapsed(s): {timetaken}")
                sys.stdout.flush()


            sents = prep.sent_detector.tokenize(fil)
            for sent in sents:

                words = tokenizer.tokenize(sent)
#                for word in words:
#                    unique_vocab.add(word)
                if len(words)>50:
                    continue
#
                input_ids = torch.tensor([tokenizer.encode(sent)])
                embeds = model(input_ids)[-2:][1][args.layer][0]
#
                compound_word = []
                w_indx = 0
#
                for w, word in enumerate(words):
                    word = word.lower()

                    if word.startswith("##"):
                        compound_word.append(word.strip('##'))
                        if w==(len(words)-1):
                            word = "".join(compound_word)
                            w2vb,w2vc = add_to(word, valid_vocab, w2vb,w2vc, embeds[w_indx])

                    else:
                        if len(compound_word)==0:
                            if w!=(len(words)-1):
                                if words[w+1].startswith('##'):
                                    compound_word.append(word.strip('##'))
                                    word_indx = w
                                    continue
                                else:
                                    w2vb,w2vc=add_to(word, valid_vocab, w2vb,
                                            w2vc,embeds[w])
                            else:
                                w2vb, w2vc = add_to(word, valid_vocab, w2vb, w2vc, embeds[w])
                                    


                        elif len(compound_word)>0:
                            if w!=(len(words)-1):
                                if words[w+1].startswith('##'):
                                    wordx = "".join(compound_word)
                                    w2vb,w2vc = add_to(wordx, valid_vocab, w2vb,w2vc, embeds[w_indx])
                                    compound_word = [word]
                                    w_indx = w

                                else:
                                    w2vb,w2vc = add_to(word, valid_vocab, w2vb,w2vc, embeds[w])
                                    word = "".join(compound_word)
                                    w2vb,w2vc = add_to(word, valid_vocab, w2vb,w2vc, embeds[w_indx])
                                    compound_word = []
                            else:
                                wordx=" ".join(compound_word)
                                w2vb, w2vc = add_to(wordx, valid_vocab, w2vb, w2vc,
                                        embeds[w_indx])

        eb_dump(i, w2vb, w2vc)
#        w2vb = {}
#        w2vc = {}


def add_to(word, valid_vocab, w2vb, w2vc, embed):

    if word in valid_vocab:
        if word in w2vb:
            w2vb[word] += embed
            w2vc[word] += 1
        else:
            w2vb[word] = embed
            w2vc[word] = 1

    return w2vb, w2vc


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

