from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import numpy as np
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

    indices = []

    for i in range(20):
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]
        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , vocab_embeddings.shape[1]))

        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = vocab_embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        center_vec = np.mean(one_cluster_tf_matrix, axis=0)

        dist_X =  np.sum((one_cluster_tf_matrix - center_vec)**2, axis = 1)

        #dist_X = np.sum(pairwise_distances(one_cluster_tf_matrix), axis = 1)
        topk = min(10, len(data_idx_within_i_cluster))
        #print(dist_X.argsort()[-topk:][::-1])

        topk_vals = dist_X.argsort()[:topk].astype(int)
        ind = []
        for i in topk_vals:
            ind.append(data_idx_within_i_cluster[i])

        indices.append(ind)

    return m_clusters, indices

def DBSCAN_model(vocab_embeddings, rand):
    dbscan = DBSCAN(eps=4.6, min_samples=5).fit(vocab_embeddings)
    m_clusters = dbscan.labels_
    print(np.unique(m_clusters))
    print(dbscan.components_.shape)
    indices = []

    for i in range(len(np.unique(m_clusters))-1):
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]
        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , vocab_embeddings.shape[1]))

        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = vocab_embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        center_vec = np.mean(one_cluster_tf_matrix, axis=0)

        dist_X =  np.sum((one_cluster_tf_matrix - center_vec)**2, axis = 1)

        #dist_X = np.sum(pairwise_distances(one_cluster_tf_matrix), axis = 1)
        topk = min(10, len(data_idx_within_i_cluster))
        #print(dist_X.argsort()[-topk:][::-1])

        topk_vals = dist_X.argsort()[:topk].astype(int)
        ind = []
        for i in topk_vals:
            ind.append(data_idx_within_i_cluster[i])

        indices.append(ind)

    return m_clusters, indices

def KMedoids_model(vocab_embeddings, topics,  rand):
    kmedoids = KMedoids(n_clusters=topics, random_state=rand).fit(vocab_embeddings)
    m_clusters = kmedoids.predict(vocab_embeddings)
    centers = np.array(kmedoids.cluster_centers_)

    indices = []

    for i in range(20):
        center_vec = centers[i]
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]
        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , centers.shape[1]))

        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = vocab_embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row


        dist_X =  np.sum((one_cluster_tf_matrix - center_vec)**2, axis = 1)
        topk = min(10, len(data_idx_within_i_cluster))
        topk_vals = dist_X.argsort()[:topk].astype(int)
        ind = []
        for i in topk_vals:
            ind.append(data_idx_within_i_cluster[i])

        indices.append(ind)

    return m_clusters, indices

def KMeans_model(vocab_embeddings, topics, vocab, rand):
    kmeans = KMeans(n_clusters=topics, random_state=rand).fit(vocab_embeddings)
    m_clusters = kmeans.predict(vocab_embeddings)
    centers = np.array(kmeans.cluster_centers_)

    indices = []

    for i in range(20):
        center_vec = centers[i]
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , centers.shape[1]))

        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = vocab_embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row


        dist_X =  np.sum((one_cluster_tf_matrix - center_vec)**2, axis = 1)
        topk = min(10, len(data_idx_within_i_cluster))
        topk_vals = dist_X.argsort()[:topk].astype(int)
        ind = []
        print(i)
        for i in topk_vals:
            ind.append(data_idx_within_i_cluster[i])
            print(vocab[data_idx_within_i_cluster[i]])

        indices.append(ind)
    return m_clusters, indices

def GMM_model(vocab_embeddings, topics, rand):
    GMM = GaussianMixture(n_components=topics, random_state=rand).fit(vocab_embeddings)
    indices = []

    for i in range(GMM.n_components):
        density = scipy.stats.multivariate_normal(cov=GMM.covariances_[i], mean=GMM.means_[i]).logpdf(vocab_embeddings)
        topk_vals = density.argsort()[-10:][::-1]
        indices.append(list(topk_vals))
    #print(indices)
    return GMM.predict(vocab_embeddings), indices, GMM

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
