import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.datasets import fetch_20newsgroups
from tensorflow.python.eager.context import eager_mode, graph_mode
import string
import random
import numpy as np



def init():
    tf.compat.v1.disable_eager_execution()
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    x = ["hello my name is ayush dalmia what is yours", "Roasted ants are a popular snack in Columbia"]
    # Extract ELMo features
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    print(embeddings.shape)

    train_data = fetch_20newsgroups(data_home="./data", subset="train", remove=('headers','footers', 'quotes'))

    files = train_data['data']

    with open('stopwords_en.txt', 'r', encoding="utf-8") as f:
        stopwords = f.readlines()

    stopwords = [s.strip() for s in stopwords]

    w2vb = {}
    w2vc = {}
    #sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

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
    print(len(valid_vocab))

    fills = []
    j = 0
    for i, fil in enumerate(files):
        if j % 10 == 0:
            fills = []
            j = 0

        fil = fil.translate(strip_punct).translate(strip_digit).lower()
        fil = ' '.join(fil.split())
        if fil == "":
            continue

        fills.append(fil)
        j+=1

        if j % 10 == 0 or i == len(files)-1 :

            embeddings = elmo(fills, signature="default", as_dict=True)["elmo"]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                embeddings = sess.run(embeddings)

            print(embeddings.shape)
            for k,fil in enumerate(fills):
                for w, word in enumerate(fil.split()):
                    if word in w2vb:
                        w2vb[word] += np.squeeze(embeddings[k, w, :])
                        w2vc[word] += 1
                    elif word in vocab_counts:
                        w2vb[word] = np.squeeze(embeddings[k, w, :])
                        w2vc[word] = 1


        if i % 10 == 0 or i == len(files)-1:
            print(i)

    #tf.compat.v1.enable_eager_execution()
    eb_dump(w2vb, w2vc)
    w2vb = {}
    w2vc = {}


def eb_dump(w2vb, w2vc):
    all_vecs = []

    for i, word in enumerate(w2vb):
        mean_vector = np.around(w2vb[word]/w2vc[word], 5)
        vect = np.append(word, mean_vector)
        if len(all_vecs)==0:
            all_vecs = vect
        else:
            all_vecs = np.vstack((all_vecs, vect))
    np.savetxt(f'models/elmo_embeddings.txt', all_vecs, fmt = '%s', delimiter=" ")
    print(len(all_vecs))


if __name__ == "__main__":
    init()
