#!/usr/bin/python3

# Standard imports
import random
import numpy as np
import pdb
import math
import os, sys
import nltk.data
import string

# custom import
import allennlp
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

# argparser
import time
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--save_fn', default="", required=False, type=str, help="filename to save elmo embeddings")
argparser.add_argument('--device', default=0, required=False)
argparser.add_argument('--data', default="20NG", required=False)
argparser.add_argument('--use_stopwords', default=0, type=int, required=False)
argparser.add_argument('--use_full_vocab', default=0, type=int, required=False)
argparser.add_argument('--mixture_coefficients', default="1;1;1", type=str, required=False, help="logits, separate by ';'")
args = argparser.parse_args()

device=torch.device("cuda:{}".format(args.device) if int(args.device)>=0 else "cpu")

print("using device:", device)


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"



class ElmoWordFromTextEncoder:

    """ Helper Class to Extract Contextualised Word Embeddings from a Document.

    1. Assumes a sentence is a window for Contextual embeddings.
    3. Extract from Elmo
    4. Requires GPU to use the encoders, not tested/debugged on cpu.

    Usage: Look at def init():

    Dependencies:
    * nltk>3.4
    * allennlp 0.9.0, pytorch > 1.2.0
    """


    def __init__(self, valid_vocab=None):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.model = Elmo(options_file, weight_file, 1, dropout=0, scalar_mix_parameters=[float(x) for x in args.mixture_coefficients.split(";")]).to(device)
        self.device = device

        self.w2vb = {} #embeds_sum
        self.w2vc = {} #counts

        if valid_vocab is None:
            print("Provide list of vocab words.")
            sys.exit(1)

        self.use_full_vocab = False
        self.strip_punct = str.maketrans("", "", string.punctuation)
        #self.strip_punct = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        self.strip_digit = str.maketrans("", "", string.digits)

        if valid_vocab is None:
            print("Provide list of vocab words.")
            sys.exit(1)
        elif valid_vocab == -1:
            self.use_full_vocab = True
            print("Extract embeddings with full vocab")

        else:
            print(f"Extract embeddings with restricted vocab, {len(valid_vocab)} words")

        self.valid_vocab = valid_vocab

    def test_encoder(self):
        #As per https://github.com/allenai/allennlp/issues/2245

        xs = ['Here is some text to encode', 'Some other text to encode']
        xs = [x.split() for x in xs]
        charids = batch_to_ids(xs).to(self.device)
        embeddings = self.model(charids)['elmo_representations'][0]
        print("Elmo models are working fine\n")


    def _add_word(self, word, emb):
        word = word.lower()
        emb = emb.cpu().detach().numpy()

        if self.use_full_vocab:
            pass
        else:
            if word not in self.valid_vocab:
                return

        if word in self.w2vb:
            self.w2vb[word] += emb
            self.w2vc[word] += 1
        else:
            self.w2vb[word] = emb
            self.w2vc[word] = 1

    def eb_dump(self, save_fn):
        print("saving embeddings")

        all_vecs = []
        for word in self.w2vb:
            mean_vector = np.around(self.w2vb[word]/self.w2vc[word], 8)
            vect = np.append(word, mean_vector)
            all_vecs.append(vect)

        np.savetxt(save_fn, np.vstack(all_vecs), fmt = '%s', delimiter=" ")
        print(f"{len(all_vecs)} vectors saved to {save_fn} ")


    def encode_docs(self, docs=[], save_fn=""):

        if len(save_fn)==0:
            save_fn = f"embeds/elmo-weighted-avg3layers.txt"
            print(f"No save filename provided, saving to: {save_fn}")

        start = time.time()
        with torch.no_grad():
            for i, doc in enumerate(docs):

                if i%(int(len(docs)/100))==0:
                    timetaken = np.round(time.time() - start, 1)
                    print(f"{i+1}/{len(docs)}, elapsed(s): {timetaken}")
                    sys.stdout.flush()

                #DB.dp()
                sents = self.sent_tokenizer.tokenize(doc) # take context to be the sentence, as in BERT
                sents = [s.translate(self.strip_punct).translate(self.strip_digit) for s in sents]
                #DB.dp()

                # do not lower case according to: https://github.com/tensorflow/hub/issues/215

                sents = [sent.split() for sent in sents]
                total_len = sum([len(s) for s in sents])

                if total_len==0:
                    continue


                while len(sents)>0:
                    sentss = sents[:50]
                    char_ids = batch_to_ids(sentss).to(self.device)
                    try:
                        embeds = self.model(char_ids)
                    except Exception as e:
                        #print("Something went wrong with:", sentss)
                        #print("Error message:", e)
                        print("Skipping..")

                        sents = sents[50:]
                        continue
                        #sys.exit(1)
                    embeds = embeds['elmo_representations'][0]

                    for s, sent in enumerate(sentss):
                        for w, word in enumerate(sentss[s]):
                            emb = np.squeeze(embeds[s, w, :])
                            self._add_word(word.lower(), emb)

                    sents = sents[50:]

        self.eb_dump(save_fn)

def init():
    """ Sample script """

    import preprocess

    if args.use_stopwords==1:
        stopwords = set(line.strip() for line in open("stopwords_en.txt"))
    else:
        stopwords = set()

    word_to_file, _, files = preprocess.create_vocab_and_files(stopwords, dataset=args.data, preprocess = 5, type="train", vocab = set())

    if args.use_full_vocab == 1:
        valid_vocab = -1
    else:
        valid_vocab = word_to_file.keys()

    encoder = ElmoWordFromTextEncoder(valid_vocab=valid_vocab)
    encoder.test_encoder()
    ### this is what you care about
    encoder.encode_docs(docs=files, save_fn=args.save_fn)


# helper function to sanity check your embeddings :/
def sanity_check(fn):
    with open(fn, 'r') as f:
        data = f.readlines()

    data = [d.split() for d in data]
    words = [d[0] for d in data]

    embeds = [np.asarray(d[1:], dtype="float") for d in data]

    for i, v in enumerate(embeds):

        maxv = max(v)
        minv = min(v)
        rangev = maxv - minv
        argmaxv = np.argmax(v)
        argminv = np.argmin(v)

        #if (maxv > 2) or (minv < -2):
        #    print(f"{words[i]} maxv: {maxv},{argmaxv} minv:{minv},{argminv}")
        if rangev > 10:
            print(f"{words[i]} range:{rangev}, maxv: {maxv},{argmaxv} minv:{minv},{argminv}")


if __name__ == "__main__":
    init()
