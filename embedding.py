import gensim
import fasttext.util
import fasttext

def create_id_dict(id2name):
    data = {}
    for line in open(id2name):
        mapping = line.split()
        data[mapping[0]] = mapping[1]
    return data

def read_entity_file(file, id_to_word):
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
        word_index[embedding[0]] = index
        index +=1
        embedding = list(map(float, embedding[1:]))
        data.append(embedding)
    print("KG: " + str(len(data)))
    return data, word_index

def find_intersect(word_index, vocab, data, type):
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

def create_entities_ft(model, train_word_to_file):
    #print("getting fasttext embeddings..")
    vocab_embeddings = []
    words = []
    for word in train_word_to_file:
        vocab_embeddings.append(model.get_word_vector(word))
        words.append(word)
    vocab_embeddings = np.array(vocab_embeddings)
    #print("complete..")
    return vocab_embeddings, words
