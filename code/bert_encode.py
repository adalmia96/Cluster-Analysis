#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Standard imports
import random
import numpy as np
import pdb
import math
import os, sys
import nltk.data
import string
import locale
locale.setlocale(locale.LC_ALL, 'en_US.utf-8')


# argparser
import time
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--nlayer', default=12, type=int, help="layer of bert to extract")
argparser.add_argument('--save_fn', default="", type=str, help="filename to save bert embeddings")
argparser.add_argument('--agg_by', default="firstword", type=str, help="method for aggregating compound words")
argparser.add_argument('--device', default=0, required=False)
argparser.add_argument('--data', default="20NG", required=False)
argparser.add_argument('--use_stopwords', default=0, type=int, required=False, help="1 for incl stopwords")
argparser.add_argument('--use_full_vocab', default=0, type=int, required=False, help="1 for incl stopwords")
args = argparser.parse_args()

# Custom imports
import torch
from pytorch_transformers import *

device = torch.device("cuda:{}".format(args.device) if int(args.device)>=0 else "cpu")
print("using device:", device)


""" Helper Class to Extract Contextualised Word Embeddings from a Document. 

1. Assumes a sentence is a window for Contextual embeddings. 
2. Deals with compound words by (1) taking the first word segment, (2) averaging word segments. 
3. Extract from Bert layer 1-12. Although people find the last layer most useful in general.
4. Requires GPU to use the transformer encoders.

Usage: Look at def init():

Dependencies:
* nltk>3.4
* pytorch 1.1.0, pytorch_transformers 1.1.0
"""


class BertWordFromTextEncoder:

    def __init__(self, valid_vocab=None):
        self.device = device
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.model, self.bert_tokenizer = self.load_bert_models()
        self.w2vb = {} #embeds_sum
        self.w2vc = {} #counts
        self.compounds = set()
        self.agg_by = ""
        self.use_full_vocab = False

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

        input_ids = torch.tensor([self.bert_tokenizer.encode('Here is some text to \
            encode')]).to(self.device)
        last_hidden_states = self.model(input_ids)[0][0]
        print("Bert models are working fine\n")


    def load_bert_models(self):
        model_class = BertModel
        tokenizer_class = BertTokenizer
        pretrained_weights = 'bert-base-uncased'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights,
                output_hidden_states=True).to(self.device)

        return model, tokenizer
    
    def _add_word(self, compound_word, compound_ixs, embeds):

        word = "".join(compound_word).lower()

        if self.agg_by=="firstword":
            w = compound_ixs[0] 
            emb = embeds[w]
        elif self.agg_by=="average":
            total_emb = 0
            for w in compound_ixs:
                total_emb += embeds[w]
            emb = total_emb/len(compound_ixs)

        emb = emb.cpu().detach().numpy()

        if self.use_full_vocab:
            pass
        else:
            if word not in self.valid_vocab:
                return

        if len(compound_ixs)>1:
            self.compounds.add(word)

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
            #word = word.encode('utf-8', 'ignore').decode('utf-8')

            mean_vector = np.around(self.w2vb[word]/self.w2vc[word], 8)
            vect = np.append(word, mean_vector)
            all_vecs.append(vect)

        #np.savetxt(f'embeds/bert_embeddings{i}-layer{args.layer}.txt', all_vecs, fmt = '%s', delimiter=" ")

        np.savetxt(save_fn, np.vstack(all_vecs), fmt = '%s', delimiter=" ", encoding='utf-8')
        print(f"{len(all_vecs)} vectors saved to {save_fn}")
        print(f"{len(self.compounds)} compound words saved to: compound_words.txt")
        
        with open('compound_words.txt', 'w') as f:
            f.write("\n".join(list(self.compounds)))

        sys.stdout.flush()

    def encode_docs(self, docs=[], agg_by="firstword", save_fn="", layer=12):
        self.agg_by = agg_by

        if len(save_fn)==0:
            save_fn = f"{args.data}-bert-layer{args.nlayer}-{agg_by}.txt"
            print(f"No save filename provided, saving to: {save_fn}")

        start = time.time()
        with torch.no_grad(): 
            for i, doc in enumerate(docs):
                if i%(int(len(docs)/100))==0:
                    timetaken = np.round(time.time() - start, 1)
                    print(f"{i+1}/{len(docs)} done, elapsed(s): {timetaken}")
                    sys.stdout.flush()
                
                # Assume a sentence as the window for contextualised embeddings.
                sents = self.sent_tokenizer.tokenize(doc)

                for sent in sents:
                    words = self.bert_tokenizer.tokenize(sent)

                    if len(words) > 500:
                        new_sents = [""]
                        fragment = ""
                        nsubwords = 0
                        currentlength = 0
                        for w in words:
                            nsubwords += 1
                            if w.startswith("##"):
                                fragment += w.replace("##", "")
                                currentlength += 1
                            else:
                                if nsubwords > 500:
                                    new_sents.append("")
                                new_sents[-1] += " " + fragment
                                fragment = w
                                currentlength = 1
                        new_sents[-1] += " " + fragment
                        new_sents = [s[1:] for s in new_sents]

                        detok = " ".join(new_sents)
                        if not words == self.bert_tokenizer.tokenize(detok):
                            pdb.set_trace()
                    else:
                        new_sents = [sent]

                    for sent in new_sents:
                        if len(new_sents) > 1:
                            words = self.bert_tokenizer.tokenize(sent)
                        try:
                            input_ids = torch.tensor([self.bert_tokenizer.encode(sent)]).to(self.device)
                        except:
                            pdb.set_trace()
                        # words correspond to input_ids correspond to embeds

                        try:
                            embeds = self.model(input_ids)[-2:][1][layer][0]
                        except Exception as e:
                            print(f"Crashed during encoding sentence: {sent}\n\n")
                            print(f"Error message:", e)
                            sys.exit(1)

                        compound_word = []
                        compound_ixs = []
                        full_word = ""

                        for w, word in enumerate(words):


                            if word.startswith('##'):
                                compound_word.append(word.replace('##',''))
                                compound_ixs.append(w)


                            else:
                                # add the previous word
                                # reset the compound word
                                if w!=0:
                                    try:
                                        self._add_word(compound_word, compound_ixs, embeds)
                                    except:
                                        pdb.set_trace()

                                compound_word = [word]
                                compound_ixs = [w]

                            if w == len(words)-1:
                                try:
                                    self._add_word(compound_word, compound_ixs, embeds)
                                except:
                                    pdb.set_trace()

        self.eb_dump(save_fn)

def init():
    """ Sample script """

    import preprocess
    if args.use_stopwords==1:
        stopwords = set(line.strip() for line in open("stopwords_en.txt", encoding='utf-8'))
    else:
        stopwords = set()

    word_to_file = {}
    word_to_file, _, files = preprocess.get_dataset(dataset=args.data, type="train")
    
    if args.use_full_vocab == 1:
        valid_vocab = -1
    else:
        valid_vocab = word_to_file.keys()

    ### this is what you care about
    encoder = BertWordFromTextEncoder(valid_vocab=valid_vocab)
    encoder.test_encoder()
    encoder.encode_docs(docs=files, save_fn=args.save_fn, agg_by=args.agg_by, layer=args.nlayer)


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
    
