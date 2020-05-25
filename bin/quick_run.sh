#!/usr/bin/env bash
# Author: Suzanna Sia

#$ -l 'hostname=b1[12345678]*|c*,mem_free=5G,ram_free=5G'
#$ -cwd
#$ -m ea
#$ -M fengsf85@gmail.com


export PYTHONIOENCODING=utf-8

source activate sphericalkmeans

EMBEDS=$1
emb=$2
ALGO=$3
weighted=$4
rr=$5

WRITEF="results/reuters/$emb-$ALGO"

PY="python3 code/score.py --entities KG --entities_file $EMBEDS/$emb --clustering_algo $ALGO --dataset reuters --vocab $EMBEDS/reuters-bert-layer12-average.full_vocab.fix --num_topics 20"

if [ "$emb" == "fasttext" ] || [ "$emb" == "word2vec" ]; then
  PY="python3 code/score.py --entities $emb --clustering_algo $ALGO --dataset reuters --vocab $EMBEDS/reuters-bert-layer12-average.full_vocab.fix --num_topics 20"
fi

if [[ $weighted -eq 1 ]]; then
  PY+=" --doc_info WGT"
  WRITEF+="-weighted"
fi

if [[ $rr -eq 1 ]]; then
  PY+=" --rerank tf"
  WRITEF+="-rr"
fi

WRITEF+=".txt"

printf "$PY > $WRITEF\n\n"
eval "$PY > $WRITEF"
