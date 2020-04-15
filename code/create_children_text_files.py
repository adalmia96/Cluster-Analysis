import string
import random
import numpy as np

def init():
    word_to_file = {}
    file_word = {}
    index = 0
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    with open('stopwords_en.txt', 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
    stopwords = [s.strip() for s in stopwords]

    for line in open('data/CBTest/data/cbt_train.txt', 'r'):
        words = line.strip()
        if "BOOK_TITLE" in words:
            continue
        elif  "CHAPTER" in words:
            words = words.lower().split()[2:]
        else:
            words = words.lower().split()


        if index % 20 == 0:
            file_word[int(index/20)] = (" ".join(words) + "\n")
        else:
            file_word[int(index/20)] += (" ".join(words)+ "\n")

        for word in words:
            word = word.translate(strip_punct)
            word = word.translate(strip_digit)

            if word in stopwords:
                continue
            if word in word_to_file:
                word_to_file[word].add(int(index/20))
            else:
                word_to_file[word]= set()
                word_to_file[word].add(int(index/20))
        index+=1

    for word in list(word_to_file):
        if len(word_to_file[word]) < 5:
            word_to_file.pop(word, None)

    for file in file_word:
        print(file_word[file])
        fil = file_word[file].translate(strip_punct).translate(strip_digit)
        file_word[file] = ""
        words = [w.strip() for w in fil.split()]
        for word in words:
            if word in word_to_file:
                file_word[file] += (word  + " ")
        print(file_word[file])
        break




    #for file in file_word:
    #    f = open(type+str(file)+".txt","w")
    #    f.write(file_word[file])
    #    f.close()

    print("Length of train files:", int(index/20))
    print("Vocab: " + str(len(word_to_file)))


if __name__ == "__main__":
    init()
