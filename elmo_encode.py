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

    for i, fil in enumerate(files):
        fil = fil.translate(strip_punct).translate(strip_digit).lower()
        fil = [' '.join(fil.split())]
        #print(fil)
        if fil[0] == "":
            continue
        embeddings = elmo(fil, signature="default", as_dict=True)["elmo"]
        for w, word in enumerate(fil[0].split()):
            if word in w2vb:
                w2vb[word] += tf.squeeze(embeddings[:, w, :])
                w2vc[word] += 1
            elif word in vocab_counts:
                w2vb[word] = tf.squeeze(embeddings[:, w, :])
                w2vc[word] = 1

        if i % 1000 == 0:
            print(i)

    #tf.compat.v1.enable_eager_execution()
    eb_dump(w2vb, w2vc)
    w2vb = {}
    w2vc = {}


def eb_dump(w2vb, w2vc):
    all_vecs = []
    with  tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for word in w2vb:
            w2vb[word] =sess.run(w2vb[word])
            mean_vector = np.around(w2vb[word]/w2vc[word], 5)
            #mean_vector = torch.mean(torch.stack(w2vb[word]), 0).detach().numpy()
            vect = np.append(word, mean_vector)

            if len(all_vecs)==0:
                all_vecs = vect
            else:
                all_vecs = np.vstack((all_vecs, vect))

    np.savetxt(f'models/elmo_embeddings.txt', all_vecs, fmt = '%s', delimiter=" ")
    print(len(all_vecs))


if __name__ == "__main__":
    init()
