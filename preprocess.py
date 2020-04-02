from sklearn.datasets import fetch_20newsgroups
import string


def create_vocab_and_files_20news(stopwords, type):
    word_to_file = {}
    train_data = fetch_20newsgroups(data_home='./data/', subset=type, remove=('headers', 'footers', 'quotes'))
    files = train_data['data'];

    #print(f"Length of {type} files:", len(files))
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    for file_num in range(0, len(files)):
        words = files[file_num].lower().split()
        for word in words:
            word = word.translate(strip_punct)
            word = word.translate(strip_digit)
            if word in stopwords:
                continue
            if word in word_to_file:
                word_to_file[word].add(file_num)
            else:
                word_to_file[word]= set()
                word_to_file[word].add(file_num)

    for word in list(word_to_file):
        if len(word_to_file[word]) < 5:
            word_to_file.pop(word, None)
    #print("Vocab: " + str(len(word_to_file)))
    return word_to_file, len(files)




def create_vocab_and_files_children(stopwords, type):

    word_to_file = {}
    index = 0
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    for line in open('data/CBTest/data/cbt_'+type+'.txt', 'r'):
        words = line.strip()
        if "BOOK_TITLE" in words:
            continue
        elif  "CHAPTER" in words:

            words = words.lower().split()[2:]
            #print(words)
        else:
            words = words.lower().split()

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

    print(f"Length of {type} files:", int(index/20))
    print("Vocab: " + str(len(word_to_file)))

    return word_to_file, index
