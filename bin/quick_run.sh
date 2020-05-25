#!/usr/bin/env bash
# Author: Suzanna Sia

#$ -l 'hostname=b1[12345678]*|c*,mem_free=5G,ram_free=5G'
#$ -cwd
#$ -m ea
#$ -M fengsf85@gmail.com


export PYTHONIOENCODING=utf-8

source activate sphericalkmeans

DS=$1
EMBEDS=$2
emb=$3
ALGO=$4
weighted=$5
rr=$6

mkdir -p results/$DS

WRITEF="results/$DS/$emb-$ALGO"

PY="python3 code/score.py --entities KG --entities_file $EMBEDS/$emb --clustering_algo $ALGO --dataset $DS --vocab $EMBEDS/$DS-bert-layer12-average.full_vocab.fix --num_topics 20"

if [ "$emb" == "fasttext" ] || [ "$emb" == "word2vec" ]; then
  PY="python3 code/score.py --entities $emb --clustering_algo $ALGO --dataset $DS --vocab $EMBEDS/$DS-bert-layer12-average.full_vocab.fix --num_topics 20"
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
