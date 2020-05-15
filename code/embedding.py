import gensim
import fasttext.util
import fasttext
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def create_id_dict(id2name):
    data = {}
    for line in open(id2name):
        mapping = line.split()
        data[mapping[0]] = mapping[1]
    return data

def read_entity_file(file, id_to_word, vocab):
    data = []
    word_index = {}
    index = 0
    mapping = None
    if id_to_word != None:
        mapping = create_id_dict(id_to_word)

    for line in open(file):
        embedding = line.split()
        if id_to_word != None:
            embedding[0] = mapping[embedding[0]][1:]
        if embedding[0] in vocab:
            word_index[embedding[0]] = index
            index +=1
            embedding = list(map(float, embedding[1:]))
            data.append(embedding)

    print("KG: " + str(len(data)))
    return data, word_index

def create_doc_to_word_emb(word_to_doc, file_num, word_list, dim):
    word_to_doc_matrix = np.zeros((len(word_list), file_num))
    for i, word in enumerate(word_list):
        for doc in word_to_doc[word]:
            word_to_doc_matrix[i][doc] += 1

    trun_ftw = TruncatedSVD(n_components=dim).fit_transform(word_to_doc_matrix)
    return trun_ftw

def find_intersect(word_index, vocab, data, files, type, add_doc):
    if add_doc == "DUP":
        return find_intersect_mult(word_index, vocab, data, type)
    elif add_doc == "SVD":
        intersection, words_index_intersect = find_intersect_unique(word_index, vocab, data, type)
        u = create_doc_to_word_emb(vocab, files, words_index_intersect, 1000)
        u = preprocessing.scale(u)
        #intersection = np.concatenate((intersection, u), axis=1)
        return u, words_index_intersect
    else:
        return find_intersect_unique(word_index, vocab, data, type)

def find_intersect_unique(word_index, vocab, data, type):
    words = []
    vocab_embeddings = []

    intersection = set(word_index.keys()) & set(vocab.keys())
    print("Intersection: " + str(len(intersection)))

    intersection = np.sort(np.array(list(intersection)))
    for word in intersection:
        if type == "word2vec":
            vocab_embeddings.append(data[word])
        else:
            vocab_embeddings.append(data[word_index[word]])
        words.append(word)

    vocab_embeddings = np.array(vocab_embeddings)

    return vocab_embeddings, words

def find_intersect_mult(word_index, vocab, data, type):
    words = []
    vocab_embeddings = []

    intersection = set(word_index.keys()) & set(vocab.keys())
    print("Intersection: " + str(len(intersection)))

    intersection = np.sort(np.array(list(intersection)))
    for word in intersection:
        for i in range(len(vocab[word])):
            if type == "word2vec":
                vocab_embeddings.append(data[word])
            else:
                vocab_embeddings.append(data[word_index[word]])
            words.append(word)
    print(len(words))
    vocab_embeddings = np.array(vocab_embeddings)
    return vocab_embeddings, words

def create_entities_ft(model, train_word_to_file, doc_info):
    #print("getting fasttext embeddings..")
    vocab_embeddings = []
    words = []
    intersection = set(train_word_to_file.keys())
    for word in intersection:
        if doc_info:
            for i in train_word_to_file[word]:
                vocab_embeddings.append(model.get_word_vector(word))
                words.append(word)
        else:
            vocab_embeddings.append(model.get_word_vector(word))
            words.append(word)
    vocab_embeddings = np.array(vocab_embeddings)
    #print("complete..")
    return vocab_embeddings, words



def get_weights_tf(vocab_list, weights):
    return np.array([len(weights[w]) for w in vocab_list])
def get_weights_tfidf(vocab_list, weights):
    return [weights[w] for w in vocab_list]
def get_weights_tfdf(vocab_list, word_file_count, files_num):
    count = np.array(get_weights_tf(vocab_list, word_file_count))
    tf = count/np.sum(count)

    df = np.array([len(np.unique(word_file_count[w])) for w in vocab_list])
    df = df/files_num

    weights = tf * df
    print(weights.shape)

    tfdf = {}
    for i, w in enumerate(vocab_list):
        tfdf[w]=weights[i]

    return weights, tfdf
def get_tfidf_score(data, train_vocab):
    tf_idf_score = {}

    tfidf_vectorizer=TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(data)

    words = tfidf_vectorizer.get_feature_names()
    total_tf_idf = tfidf_vectorizer_vectors.toarray().sum(axis=0)

    vocab = set(words) & set(train_vocab.keys())
    for i, word in enumerate(words):
        if word in vocab:
            tf_idf_score[word] = total_tf_idf[i]

    return tf_idf_score
