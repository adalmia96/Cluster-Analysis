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

    vocab = create_global_vocab(args.vocab)
    train_word_to_file, train_w_to_f_mult, files = create_vocab_and_files(stopwords, args.dataset, args.preprocess, "train", vocab)
    files_num = len(files)

    intersection = None
    words_index_intersect = None


    tf_idf = get_tfidf_score(files, train_word_to_file)


    if args.entities == "word2vec":
        model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        intersection, words_index_intersect  = find_intersect(model.vocab,  train_w_to_f_mult, model, files_num, args.entities, args.doc_info)
    elif args.entities == "fasttext":
        ft = fasttext.load_model('models/wiki.en.bin')
        intersection, words_index_intersect = create_entities_ft(ft, train_w_to_f_mult, args.doc_info)
        print(intersection.shape)
    elif args.entities == "KG":
        data, word_index = read_entity_file(args.entities_file, args.id2name, train_word_to_file)
        intersection, words_index_intersect = find_intersect(word_index, train_w_to_f_mult, data, files_num, args.entities, args.doc_info)




    if args.use_dims:
        intersection = PCA_dim_reduction(intersection, args.use_dims)

    #weights , tfdf = get_weights_tfdf(words_index_intersect, train_w_to_f_mult, files_num)
    #weights = None

    if args.doc_info == "WGT":
        weights  = get_weights_tf(words_index_intersect, train_w_to_f_mult)


    test_word_to_file, test_word_to_file_mult, test_files = create_vocab_and_files(stopwords, args.dataset,args.preprocess, "test", vocab)
    test_files_num = len(test_files)

    npmis = []
    labels = None
    top_k = None
    pmi_mat = None
    #pmi_mat = calc_pmi_matrix(words_index_intersect, train_word_to_file, files_num)

    #eps = np.arange(4.73, 4.75, 0.005)
    for rand in range(NSEEDS):
        #Distance Based
        if args.clustering_algo == "KMeans":
            labels, top_k  = KMeans_model(intersection, words_index_intersect, args.num_topics, args.rerank, rand, weights)
        elif args.clustering_algo == "SPKMeans":
            labels, top_k  = SphericalKMeans_model(intersection, words_index_intersect, args.num_topics, args.rerank, rand, weights)
        elif args.clustering_algo == "GMM":
            labels, top_k = GMM_model(intersection, words_index_intersect, args.num_topics, args.rerank, rand)
        elif args.clustering_algo == "KMedoids":
            labels, top_k  = KMedoids_model(intersection,  words_index_intersect,  args.num_topics, rand)
        elif args.clustering_algo == "VMFM":
            labels, top_k = VonMisesFisherMixture_Model(intersection, args.num_topics, rand)

        #Affinity matrix based
        elif args.clustering_algo == "DBSCAN":
            k=5.8
            labels, top_k  = DBSCAN_model(intersection,k)
        elif args.clustering_algo == "Agglo":
            labels, top_k  = Agglo_model(intersection, args.num_topics, rand)
        elif args.clustering_algo == "Spectral":
            labels, top_k  = SpectralClustering_Model(intersection,args.num_topics, rand,  pmi_mat)

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

        else:
            bins, top_k_words = sort(labels, top_k,  words_index_intersect)
            if args.rerank=="freq":
                top_k_words =  rank_freq(top_k_words, train_w_to_f_mult)
            elif args.rerank=="tfidf":
                top_k_words = rank_td_idf(top_k_words, tf_idf)
            elif args.rerank=="tfdf":
                top_k_words = rank_td_idf(top_k_words, tfdf)

        val = npmi.average_npmi_topics(top_k_words, len(top_k_words), test_word_to_file,
                test_files_num)

        npmi_score = np.around(val, 5)
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
    parser.add_argument("--entities", type=str, choices=["word2vec", "fasttext", "KG"])
    parser.add_argument( "--entities_file", type=str, help="entity file")

    parser.add_argument("--clustering_algo", type=str, required=True, choices=["KMeans", "SPKMeans", "GMM", "KMedoids","Agglo","DBSCAN","Spectral","VMFM",
        'from_file', 'LDA'])

    parser.add_argument( "--topics_file", type=str, help="topics file")

    parser.add_argument('--use_dims', type=int)
    parser.add_argument('--num_topics', type=int, default=20)
    parser.add_argument("--doc_info", type=str, choices=["SVD", "DUP", "WGT"])
    parser.add_argument("--rerank", type=str, choices=["tf", "tfidf", "tfdf"])

    parser.add_argument('--id2name', type=Path, help="id2name file")

    parser.add_argument("--dataset", type=str, required=True, choices=["fetch20", "children", "reuters"])
    parser.add_argument("--preprocess", type=bool, default=False)
    parser.add_argument("--vocab", required=True,  nargs='+', default=[])


    args = parser.parse_args()
    return args



if __name__ == "__main__":
    main()
