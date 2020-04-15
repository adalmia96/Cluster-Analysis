from clustering import *
from preprocess import *
from embedding import *

from sklearn.metrics import pairwise_distances_argmin_min
import sys
import npmi
import argparse
import string
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pdb
import math

NSEEDS = 5

def main():
    args = parse_args()

    stopwords = set(line.strip() for line in open('stopwords_en.txt'))
    train_word_to_file, train_w_to_f_mult, files_num = create_vocab_and_files_20news(stopwords, "train")
    #train_word_to_file, train_w_to_f_mult, files_num = create_vocab_and_files_children(stopwords, "train")
    intersection = None
    words_index_intersect = None

    if args.entities == "word2vec":
        model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        intersection, words_index_intersect  = find_intersect(model.vocab,  train_w_to_f_mult, model, files_num, args.entities, args.doc_info)
    elif args.entities == "fasttext":
        ft = fasttext.load_model('models/wiki.en.bin')
        intersection, words_index_intersect = create_entities_ft(ft, train_word_to_file)
        print(intersection.shape)

    elif args.entities == "KG":
        data, word_index = read_entity_file(args.entities_file, args.id2name)
        intersection, words_index_intersect = find_intersect(word_index, train_w_to_f_mult, data, files_num, args.entities, args.doc_info)

    if args.use_dims:
        intersection = PCA_dim_reduction(intersection, args.use_dims)
        #intersection = TSNE_dim_reduction(intersection, args.use_dims)

    test_word_to_file, test_word_to_file_mult, test_files_num = create_vocab_and_files_20news(stopwords, "test")

    #test_word_to_file, test_word_to_file_mult, test_files_num = create_vocab_and_files_children(stopwords, "combined")

    npmis = []
    labels = None
    top_k = None
    gmm = None
    n_p = None
    pmi_mat = None
    #pmi_mat = calc_pmi_matrix(words_index_intersect, train_word_to_file, files_num)
    #eps = np.arange(4.73, 4.75, 0.005)
    for rand in range(NSEEDS):
        #print("Eps:" + str(rand))
        if args.clustering_algo == "KMeans":
            labels, top_k  = KMeans_model(intersection, words_index_intersect, args.topics, rand)
        elif args.clustering_algo == "SPKMeans":
            labels, top_k  = SphericalKMeans_model(intersection, args.topics, rand)
        elif args.clustering_algo == "Spectral":
            labels, top_k  = SpectralClustering_Model(intersection, args.topics, rand,  pmi_mat)
        elif args.clustering_algo == "KMedoids":
            labels, top_k  = KMedoids_model(intersection,  words_index_intersect, args.topics, rand)
        elif args.clustering_algo == "Agglo":
            labels, top_k  = Agglo_model(intersection, args.topics, rand)
        elif args.clustering_algo == "DBSCAN":
            print(k)
            labels, top_k  = DBSCAN_model(intersection_unique,  k)
        elif args.clustering_algo == "GMM":
            # top_k are indexes of the vocabulary
            labels, top_k, gmm  = GMM_model(intersection, words_index_intersect, args.topics, rand)
        elif args.clustering_algo == "VMFM":
            # top_k are indexes of the vocabulary
            labels, top_k = VonMisesFisherMixture_Model(intersection, args.topics, rand)

        if args.clustering_algo == 'from_file':
            with open('bert_topics.txt', 'r') as f:
                top_k_words = f.readlines()
            top_k_words = [tw.strip().replace(',', '').split() for tw in top_k_words]
        elif args.clustering_algo == 'LDA':
            with open(args.topics_file, 'r') as f:
                top_k_words = f.readlines()
            top_k_words = [tw.strip().replace(',', '').split() for tw in top_k_words]
            for i, top_k in enumerate(top_k_words):
                top_k_words[i] = top_k_words[i][2:12]
            #print(top_k_words)
        else:
            bins, top_k_words = sort(labels, top_k,  words_index_intersect)
                #print(top_k_words)
            # don't overload function name.
        #val, n_p = get_npmi(top_k_words, test_word_to_file, test_files_num)

        val2 = npmi.average_npmi_topics(top_k_words, len(top_k_words), test_word_to_file,
                test_files_num)

        npmi_score = np.around(val2, 5)
        print("NPMI:" + str(npmi_score))
        npmis.append(npmi_score)
            #break;
            #break;
            #with open(f'{args.entities_file}_npmi.txt', 'a') as f:
            #    f.write(f'{rand}\t{args.clustering_algo}\t{args.use_dims}\t{npmi_score}\n')
    print("NPMI Mean:" + str(np.mean(npmis)))


def sort(labels, indices, word_index):
    bins = {}
    index = 0
    top_k_bins = []
    for label in labels:
        if label not in bins:
            bins[label] = [word_index[index]]
        else:
            bins[label].append(word_index[index])
        index += 1;
    for i in range(0, len(indices)):
        ind = indices[i]
        top_k = []
        for word_ind in ind:
            top_k.append(word_index[word_ind])
        top_k_bins.append(top_k)
    return bins, top_k_bins
def print_bins(bins, name, type):
    f = open(name + "_" + type + "_corpus_bins.txt","w+")
    for i in range(0, 20):
        f.write("Bin " + str(i) + ":\n")
        for word in bins[i]:
            f.write(word + ", ")
        f.write("\n\n")

    f.close()
def print_top_k(top_k_bins, name, type):
    f = open(name + "_" + type + "_corpus_top_k.txt","w+")
    for i in range(0, 20):
        f.write("Bin " + str(i) + ":\n")
        top_k = top_k_bins[i]
        for word in top_k:
            f.write(word + ", ")
        f.write("\n\n")
    f.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--entities", type=str, required=True, choices=["word2vec", "fasttext", "KG"])
    parser.add_argument("--clustering_algo", type=str, required=True, choices=["KMeans", "SPKMeans", "GMM", "KMedoids","Agglo","DBSCAN","Spectral","VMFM",
        'from_file', 'LDA'])
    parser.add_argument( "--entities_file", type=str, help="entity file")
    parser.add_argument( "--topics_file", type=str, help="topics file")
    parser.add_argument('--id2name', type=Path, help="id2name file")
    parser.add_argument('--use_dims', type=int, default=300)
    parser.add_argument('--topics', type=int, default=20)

    parser.add_argument("--doc_info", type=str, choices=["SVD", "DUP"])
    args = parser.parse_args()
    return args

def calc_pmi_matrix(word_intersect, word_in_file, window_total):
    pmi = np.zeros((len(word_intersect), len(word_intersect)))
    for i in range(len(word_intersect)):
        for j in range(i, len(word_intersect)):
            pmi[i, j] = pmi_wpair(word_intersect[i], word_intersect[j], word_in_file, window_total)
            pmi[j, i] = pmi[i, j]
    print(pmi)
    return pmi

"""
Deprecated; See npmi.py instead
def pmi_wpair(word1, word2, word_in_file, window_total):
    eps = 10**(-12)
    w1_count = 0
    w2_count = 0
    combined_count = 0
    if word1 in word_in_file and word2 in word_in_file:
        combined_count = len(set(word_in_file[word1]) & set(word_in_file[word2]))
        w1_count = len(word_in_file.get(word1, []))
        w2_count = len(word_in_file.get(word2, []))
    result = np.log(((float(combined_count)*float(window_total)) + eps)/ \
                (float(w1_count*w2_count)+eps))
    return result


def npmi_wpair(word1, word2, word_in_file, window_total):
    eps = 10**(-12)
    w1_count = 0
    w2_count = 0
    combined_count = 0
    if word1 in word_in_file and word2 in word_in_file:
        combined_count = len(set(word_in_file[word1]) & set(word_in_file[word2]))
        w1_count = len(word_in_file.get(word1, []))
        w2_count = len(word_in_file.get(word2, []))
    result = np.log(((float(combined_count)*float(window_total)) + eps)/ \
                (float(w1_count*w2_count)+eps))
    result = result / (-1.0*np.log(float(combined_count)/(window_total) + eps))
    return result

def calc_topic_coherence(topic_words, word_in_file, files_num):
    topic_assoc = []
    for i in range(0, len(topic_words)-1):
        w1 = topic_words[i]
        for j in range(i+1, len(topic_words)):
            w2 = topic_words[j]
            #print(w1 + " " + w2 + str(npmi_wpair(w1, w2, word_in_file, files_num)))
    #        if w1 != w2:
            topic_assoc.append(npmi_wpair(w1, w2, word_in_file, files_num))
    if len(topic_assoc)==0:
        pdb.set_trace()
    return float(sum(topic_assoc))/len(topic_assoc)

def get_npmi(top_k_bins, word_in_file, files_num):
    ntopics = len(top_k_bins)
    npmi_scores = np.zeros(ntopics)
    for k in range(ntopics):
        npmi_score = calc_topic_coherence(top_k_bins[k], word_in_file, files_num)
        print(np.around(npmi_score, 5), " ".join(top_k_bins[k]))
        npmi_scores[k] = np.around(npmi_score, 5)
    return np.mean(npmi_scores), npmi_scores
"""


if __name__ == "__main__":
    main()
