words = []
data = []
for line in open("models/bert_embeddings11313.txt"):
    embedding = line.split()
    words.append(embedding[0])
    embedding = list(map(float, embedding[1:]))
    data.append(embedding)
f = open("bert_visualize.tsv","w+")
for i in range(len(data)):
    for i, pnt in enumerate(data[i]):
        if i == len(data[i]) -1:
            f.write(str(pnt)+ '\n')
        else:
            f.write(str(pnt)+ '\t')

f = open("bert_visualize_words.tsv","w+")
for i in range(len(words)):
        f.write(words[i]+ '\n')
