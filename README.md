# Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too!
The repo contains the source code for the paper: link here

## How to use the code
To cluster the word embeddings to discover the latent topics, run the code/score.py file.
Here are the arguments that can be passed in:

### Required:
`--entities` : The type of pre-trained word embedding you are clustering with\
choices= word2vec, fasttext, glove, KG 
KG stands for your own set of embeddings 

`--entities_file`: The file name contain the embeddings 

`--clustering_algo`: The clustering algorithm to use  
choices= KMeans, SPKMeans, GMM, KMedoids, Agglo, DBSCAN , Spectral, VMFM

`--vocab`: List of vocab files to use for tokenization 

### Not Required:
`--dataset`: Dataset to test clusters against against\
default = 20NG 
choices= 20NG, reuters

`--preprocess`: Cuttoff threshold for words to keep in the vocab based on frequency 

`--use_dims`: Dimensions to scale with PCA (much be less than orginal dims)

`--num_topics`: List of number of topics to try 
default: 20

`--doc_info`: How to add document information
 choices= DUP, WGT
 
`--rerank`: Value used for reranking the words in a cluster  
choices=tf, tfidf, tfdf

Example call:
`python3 code/score.py --entities KG --entities_file {dest_to_entities_file} --clustering_algo GMM --dataset reuters --vocab {dest_to_vocab_file} --num_topics 20 50 --doc_info WGT--rerank tf`
