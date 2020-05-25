from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
#from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import pdb

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture

from sklearn.metrics.pairwise import rbf_kernel

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import networkx as nx
import scipy.stats

def PCA_dim_reduction(intersection, dim):
    intersection = intersection - np.mean(intersection, axis = 0)
    sigma = np.cov(intersection.T)
    eigVals, eigVec = np.linalg.eig(sigma)
    sorted_index = eigVals.argsort()[::-1]
    eigVals = eigVals[sorted_index]
    eigVec = eigVec[:,sorted_index]
    eigVec = eigVec[:,:dim]
    transformed = intersection.dot(eigVec)
    return transformed

def TSNE_dim_reduction(intersection, dim):
    X_embedded = TSNE(n_components=dim).fit_transform(intersection)
    return X_embedded

def Agglo_model(vocab_embeddings, topics, rand):
    agglo = AgglomerativeClustering(n_clusters=topics).fit(vocab_embeddings)
    m_clusters = agglo.labels_
    return m_clusters, find_words_for_cluster(m_clusters, topics)

def DBSCAN_model(vocab_embeddings, e=0.5):
    dbscan = DBSCAN(eps=e, min_samples=10).fit(vocab_embeddings)
    m_clusters = dbscan.labels_
    clusters = len(np.unique(m_clusters[m_clusters>= 0]))
    return m_clusters, find_words_for_cluster(m_clusters, clusters)

def SpectralClustering_Model(vocab_embeddings, topics, rand, pmi):
    precomp = rbf_kernel(vocab_embeddings)

    #print(precomp)
    #pmax, pmin = pmi.max(), pmi.min()
    #pmi = (pmi - pmin)/(pmax - pmin)

    #precomp = precomp * pmi
    #print(precomp)

    #nearest_neighbors
    #precomputed
    SC = SpectralClustering(n_clusters=topics, random_state=rand, affinity = "nearest_neighbors").fit(vocab_embeddings)
    m_clusters = SC.labels_

    return m_clusters, find_words_for_cluster(m_clusters, topics)

def KMedoids_model(vocab_embeddings, vocab, topics,  rand):
    kmedoids = KMedoids(n_clusters=topics, random_state=rand).fit(vocab_embeddings)
    m_clusters = kmedoids.predict(vocab_embeddings)
    centers = np.array(kmedoids.cluster_centers_)
    indices = []

    for i in range(20):
        topk_vals = sort_closest_center(centers[i], m_clusters, vocab_embeddings, i)
        indices.append(find_top_k_words(100, topk_vals, vocab))

    return m_clusters, indices

def KMeans_model(vocab_embeddings, vocab, topics, rerank, rand, weights):
    kmeans = KMeans(n_clusters=topics, random_state=rand).fit(vocab_embeddings, sample_weight=weights)
    m_clusters = kmeans.predict(vocab_embeddings, sample_weight=weights)
    centers = np.array(kmeans.cluster_centers_)

    indices = []

    for i in range(topics):
        topk_vals = sort_closest_center(centers[i], m_clusters, vocab_embeddings, i)
        if rerank:
            indices.append(find_top_k_words(100, topk_vals, vocab))
        else:
            indices.append(find_top_k_words(10, topk_vals, vocab))
        #print(indices)
    return m_clusters, indices


def SphericalKMeans_model(vocab_embeddings,vocab,topics, rerank, rand, weights):

    spkmeans = SphericalKMeans(n_clusters=topics, random_state=rand).fit(vocab_embeddings, sample_weight=weights)
    m_clusters = spkmeans.predict(vocab_embeddings,  sample_weight=weights)
    centers = np.array(spkmeans.cluster_centers_)

    indices = []

    for i in range(topics):
        topk_vals = sort_closest_cossine_center(centers[i], m_clusters, vocab_embeddings, i)
        if rerank:
            indices.append(find_top_k_words(100, topk_vals, vocab))
        else:
            indices.append(find_top_k_words(10, topk_vals, vocab))
        #print(indices)
    return m_clusters, indices

def GMM_model(vocab_embeddings, vocab,  topics, rerank, rand):
    GMM = GaussianMixture(n_components=topics, random_state=rand).fit(vocab_embeddings)
    indices = []
    for i in range(GMM.n_components):
        density = scipy.stats.multivariate_normal(cov=GMM.covariances_[i], mean=GMM.means_[i]).logpdf(vocab_embeddings)
        topk_vals = density.argsort()[-1*len(density):][::-1].astype(int)
        if rerank:
            indices.append(find_top_k_words(100, topk_vals, vocab))
        else:
            indices.append(find_top_k_words(10, topk_vals, vocab))

    return GMM.predict(vocab_embeddings), indices

def VonMisesFisherMixture_Model(vocab_embeddings, vocab, topics, rerank, rand):
    #vmf_soft = VonMisesFisherMixture(n_clusters=topics, posterior_type='hard', n_jobs=-1, random_state=rand).fit(vocab_embeddings)
    print("fitting vmf...")
    vmf_soft = VonMisesFisherMixture(n_clusters=topics, posterior_type='soft', n_jobs=-1, random_state=rand).fit(vocab_embeddings)

    llh = vmf_soft.log_likelihood(vocab_embeddings)
    indices = []
    for i in range(topics):

        topk_vals = llh[i, :].argsort()[::-1].astype(int)
        if rerank:
            indices.append(find_top_k_words(100, topk_vals, vocab))
        else:
            indices.append(find_top_k_words(10, topk_vals, vocab))

    return vmf_soft.predict(vocab_embeddings), indices

def sort_closest_center(center_vec, m_clusters,vocab_embeddings, c_ind):
    data_idx_within_i_cluster = np.array([ idx for idx, clu_num in enumerate(m_clusters) if clu_num == c_ind ])
    one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , center_vec.shape[0]))

    for row_num, data_idx in enumerate(data_idx_within_i_cluster):
        one_row = vocab_embeddings[data_idx]
        one_cluster_tf_matrix[row_num] = one_row

    dist_X =  np.sum((one_cluster_tf_matrix - center_vec)**2, axis = 1)
    #topk = min(10, len(data_idx_within_i_cluster))
    #topk_vals = dist_X.argsort()[:topk].astype(int)

    topk_vals = dist_X.argsort().astype(int)
    topk_vals = data_idx_within_i_cluster[topk_vals]

    return topk_vals

def sort_closest_cossine_center(center_vec, m_clusters,vocab_embeddings, c_ind):
        data_idx_within_i_cluster = np.array([ idx for idx, clu_num in enumerate(m_clusters) if clu_num == c_ind ])
        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , center_vec.shape[0]))

        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = vocab_embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        dist_X =  (cosine_similarity(one_cluster_tf_matrix, center_vec.reshape(1, -1))).squeeze()
        dist_X = 2.0*(1.0-dist_X)
        #topk = min(10, len(data_idx_within_i_cluster))
        #topk_vals = dist_X.argsort()[:topk].astype(int)

        topk_vals = dist_X.argsort().astype(int)
        topk_vals = data_idx_within_i_cluster[topk_vals]

        return topk_vals

def find_top_k_words(k, top_vals, vocab):
    ind = []
    unique = set()
    for i in top_vals:
        word = vocab[i]
        if word not in unique:
            ind.append(i)
            unique.add(vocab[i])
            if len(unique) == k:
                break
    return ind



def rank_freq(top_k_words, train_w_to_f_mult):
    top_10_words = []
    for words in top_k_words:
        words = np.array(words)
        count = np.array([len(train_w_to_f_mult[word]) for word in words ])
        topk_vals = count.argsort()[-10:][::-1].astype(int)
        top_10_words.append(words[topk_vals])
    return top_10_words

def rank_td_idf(top_k_words, tf_idf):
    top_10_words = []
    for words in top_k_words:
        words = np.array(words)
        count = np.array([tf_idf[word] for word in words ])
        #topk_vals = count.argsort()[-10:][::-1].astype(int)
        topk_vals = count.argsort()[-10:][::-1].astype(int)
        top_10_words.append(words[topk_vals])
    return top_10_words


def rank_centrality(top_k_words, top_k, word_in_file):
    for i, cluster in enumerate(top_k):
        cluster = np.array(cluster)

        subgraph = calc_coo_matrix(top_k_words[i], word_in_file)
        G = nx.from_numpy_matrix(subgraph)
        sc = nx.subgraph_centrality(G)

        ind = np.argsort([sc[node] for node in sorted(sc)])[-10:][::-1].astype(int)


        top_k_words[i] = np.array(top_k_words[i])[ind]
    return top_k_words


def calc_coo_matrix(word_intersect, word_in_file):
    coo = np.zeros((len(word_intersect), len(word_intersect)))
    for i in range(len(word_intersect)):
        for j in range(i, len(word_intersect)):
            coo[i, j] = count_wpair(word_intersect[i], word_intersect[j], word_in_file)
            coo[j, i] = coo[i, j]
    return coo

def count_wpair(word1, word2, word_in_file):
    combined_count = 0
    if word1 != word2:
        combined_count = len(set(word_in_file[word1]) & set(word_in_file[word2]))
    return combined_count


def find_words_for_cluster(m_clusters,  clusters):
    indices = []
    for i in range(0, clusters):
        if i == -1:
            continue
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]
        indices.append(data_idx_within_i_cluster)
    return indices

def visualize(intersection):
    intersection_red = TSNE_dim_reduction(intersection, 2)
    for i in range(0,len(n_p)):
        labels = np.where(labels==i, n_p[i], labels)
    plt.scatter(intersection_red[:, 0], intersection_red[:, 1], c=labels, vmin=-0.5, vmax=0.5,  s=5, cmap='RdBu')

    centers = np.empty(shape=(gmm.n_components, intersection_red.shape[1]))
    for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(intersection)
        centers[i, :] = intersection_red[np.argmax(density)]

    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=35, alpha=0.7)
    plt.show(block=True)
