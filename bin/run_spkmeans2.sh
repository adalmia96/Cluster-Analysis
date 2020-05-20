#!/usr/bin/env bash
#Author: Suzanna Sia
#EMBED_DIR=/export/c12/ssia/shared/Cluster-Analysis/embeds
EMBED_DIR=embeds
algo=KMeans

mkdir -p results || exit "Error mkdir"

#for em in glove.840B.300d.txt; do
#for em in 20NG-bert-layer12-average.txt 20NG-elmo-weighted-avg3layers.txt; do
for em in 20ng-elmo-layer3.txt.swr; do
#for em in jose_300d.txt bert_embeddings-layer12-firstword.txt 20NG-elmo-weighted-avg3layers.txt bert_embeddings-layer12-average.txt; do

  echo "running weighted rr, $em"
  python3 code/score.py --entities KG --entities_file ${EMBED_DIR}/${em} --clustering_algo $algo --doc_info WGT --rerank freq > results/$em-$algo-weighted-rr.txt

  echo "running weighted, $em"
  python3 code/score.py --entities KG --entities_file ${EMBED_DIR}/${em} --clustering_algo $algo --doc_info WGT > results/$em-$algo-weighted.txt

  echo "running rr, $em"
  python3 code/score.py --entities KG --entities_file ${EMBED_DIR}/${em} --clustering_algo $algo --rerank freq > results/$em-$algo-rr.txt

  echo "running normal, $em"
  python3 code/score.py --entities KG --entities_file ${EMBED_DIR}/${em} --clustering_algo $algo > results/$em-$algo.txt

done
