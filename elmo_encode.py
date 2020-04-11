import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
from sklearn.datasets import fetch_20newsgroups
import string
import random
import numpy as np



def init():
    tf.disable_eager_execution()
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





if __name__ == "__main__":
    init()
