Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too! (2020; Code for paper)
==============================

The repo contains the code needed to reproduce the results in [Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too!]( https://aclanthology.org/2020.emnlp-main.135.pdf) by Sia, Dalmia, and Mieke (2020)

Sia, S., Dalmia, A., & Mielke, S. J. (2020). Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too! Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1728–1736. https://doi.org/10.18653/v1/2020.emnlp-main.135


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

## How to cite
``` bibtex
@inproceedings{sia-etal-2020-tired,
    title = "Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too!",
    author = "Sia, Suzanna  and
      Dalmia, Ayush  and
      Mielke, Sabrina J.",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.135",
    doi = "10.18653/v1/2020.emnlp-main.135",
    pages = "1728--1736",
}
```
