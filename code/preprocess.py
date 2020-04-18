from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold
import string
import numpy as np


def combine_split_children():
    files = []
    index = 0
    with open('data/CBTest/data/cbt_train.txt') as fp:
        data = fp.readlines()
    with open('data/CBTest/data/cbt_valid.txt') as fp:
        data2 = fp.readlines()
    with open('data/CBTest/data/cbt_test.txt') as fp:
        data3 = fp.readlines()
    data += "\n"
    data += data2
    data += "\n"
    data += data3

    for line in data:
        words = line.strip()
        if "BOOK_TITLE" in words:
            continue
        elif  "CHAPTER" in words:
            words = words.split()[2:]
        else:
            words = words.split()

        if "-RRB-" in words:
            words.remove("-RRB-")
        if "-LRB-" in words:
            words.remove("-LRB-")

        sentence = (" ".join(words) + "\n")
        if "-RCB-" in words:
             sentence = sentence[0:sentence.find("-")] + sentence[sentence.rfind("-")+1:]
             #print(sentence)

        if index % 20 == 0:
            files.append(sentence)
        else:
            files[int(index/20)] += sentence

        index += 1
    files = np.array(files)


    kf = KFold(n_splits=10, shuffle=True, random_state = 0)
    indices = list(kf.split(files))[0]

    train = files[indices[0]]
    test = files[indices[1]]

    kf = KFold(n_splits=9, shuffle=True, random_state = 0)
    indices = list(kf.split(train))[0]

    train = files[indices[0]]
    valid = files[indices[1]]

    return train, valid, test
    # print(files)



def create_vocab_and_files_20news(stopwords, type):
    word_to_file = {}
    word_to_file_mult = {}

    train_data = fetch_20newsgroups(data_home='./data/', subset=type, remove=('headers', 'footers', 'quotes'))
    files = train_data['data'];
    #doc_to_word = np.zeros
    return create_vocab(stopwords, files)





def create_vocab(stopwords, data):
    word_to_file = {}
    word_to_file_mult = {}
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    for file_num in range(0, len(data)):
        words = data[file_num].lower().split()
        #words = [w.strip() for w in words]
        for word in words:
            if "@" in word and "." in word:
                continue
            word = word.translate(strip_punct)
            word = word.translate(strip_digit)
            if word in stopwords:
                continue

            if word in word_to_file:
                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)
            else:
                word_to_file[word]= set()
                word_to_file_mult[word] = []

                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)

    for word in list(word_to_file):
        if len(word_to_file[word]) <= 5  or len(word) <= 3:
            word_to_file.pop(word, None)
            word_to_file_mult.pop(word, None)
    print("although" in word_to_file)
    print("Files:" + str(len(data)))
    print("Vocab: " + str(len(word_to_file)))

    return word_to_file, word_to_file_mult, data






def create_vocab_and_files_children(stopwords, type):

    word_to_file = {}
    word_to_file_mult = {}
    index = 0
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    for line in open('data/CBTest/data/cbt_'+type+'.txt', 'r'):
        words = line.strip()
        if "BOOK_TITLE" in words:
            continue
        elif  "CHAPTER" in words:
            words = words.lower().split()[2:]
        else:
            words = words.lower().split()


        for word in words:
            word = word.translate(strip_punct)
            word = word.translate(strip_digit)

            if word in stopwords:
                continue
            if word in word_to_file:
                word_to_file[word].add(int(index/20))
                word_to_file_mult[word].append(int(index/20))
            else:
                word_to_file[word]= set()
                word_to_file_mult[word] = []

                word_to_file[word].add(int(index/20))
                word_to_file_mult[word].append(int(index/20))
        index+=1

    for word in list(word_to_file):
        if len(word_to_file[word]) < 5 or len(word) <= 3:
            word_to_file.pop(word, None)
            word_to_file_mult.pop(word, None)
    print(f"Length of {type} files:", int(index/20))
    print("Vocab: " + str(len(word_to_file)))

    return word_to_file, word_to_file_mult, index
