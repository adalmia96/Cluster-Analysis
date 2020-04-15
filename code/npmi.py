#!/usr/bin/python3
# Author: Suzanna Sia
# Credits: Ayush Dalmia

# Standard imports
#import random
import numpy as np
#import pdb
import math
import os, sys

# argparser
import argparse
import pdb
#from distutils.util import str2bool

# Custom imports

def average_npmi_topics(topic_words, ntopics, word_doc_counts, nfiles):

    eps = 10**(-12)

    all_topics = []
    for k in range(ntopics):
        word_pair_counts = 0
        topic_score = []

        ntopw = len(topic_words[k])

        for i in range(ntopw-1):
            for j in range(i+1, ntopw):
                w1 = topic_words[k][i]
                w2 = topic_words[k][j]

                w1w2_dc = len(word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set()))
                w1_dc = len(word_doc_counts.get(w1, set()))
                w2_dc = len(word_doc_counts.get(w2, set()))

                # what we had previously:
                #pmi_w1w2 = np.log(((w1w2_dc * nfiles) + eps) / ((w1_dc * w2_dc) + eps))

                # Correct eps:
                pmi_w1w2 = np.log((w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps)
                npmi_w1w2 = pmi_w1w2 / (- np.log( (w1w2_dc)/nfiles + eps))

                # Sanity check Which is equivalent to this:
                #if w1w2_dc ==0:
                #    npmi_w1w2 = -1
                #else:
                    #pmi_w1w2 = np.log( (w1w2_dc * nfiles)/ (w1_dc*w2_dc))
                    #npmi_w1w2 = pmi_w1w2 / (-np.log(w1w2_dc/nfiles))

                #if npmi_w1w2>1 or npmi_w1w2<-1:
                #    print("NPMI score not bounded for:", w1, w2)
                #    print(npmi_w1w2)
                #    sys.exit(1)

                topic_score.append(npmi_w1w2)

        all_topics.append(np.mean(topic_score))

    for k in range(ntopics):
        print(np.around(all_topics[k],5), " ".join(topic_words[k]))

    avg_score = np.around(np.mean(all_topics), 5)
    print(f"\nAverage NPMI for {ntopics} topics: {avg_score}")

    return avg_score
