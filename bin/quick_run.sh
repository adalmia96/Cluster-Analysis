#!/usr/bin/env bash

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
#scale=$7

mkdir -p results/${DS}_nofix

WRITEF="results/${DS}_nofix/$emb-$ALGO"

PY="python3 code/score.py --clustering_algo $ALGO --dataset $DS --vocab $EMBEDS/${DS}-bert-layer12-average.full_vocab --num_topics 20"

# glove needs both --entities and --entities_file flag
# fasttext and word2vec only take --entities flag
# the rest take --entities KG and --entities_file flag

if [ "$emb" == "glove" ]; then
  PY+=" --entities glove --entities_file $EMBEDS/glove.840B.300d.txt"

elif [ "$emb" == "fasttext" ] || [ "$emb" == "word2vec" ]; then
  PY+=" --entities $emb"

else
  PY+=" --entities KG --entities_file $EMBEDS/$emb"

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
#eval "$PY"
