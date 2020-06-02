#!/usr/bin/env bash

CUDA=`free-gpu`
export CUDA_VISIBLE_DEVICES=$CUDA
CUDA=0

SAVEDIR=/export/c12/ssia/shared/Cluster-Analysis/embeds

DATA=cb
layer=12
use_sw=0
use_full_vocab=1
agg_by=average

for DATA in cb reuters; do
#for DATA in 20NG; do

  SAVEFN=$SAVEDIR/${DATA}-bert-layer${layer}-${agg_by}.full_vocab.fix

  python code/bert_encode.py --nlayer $layer --device $CUDA --data $DATA --save_fn $SAVEFN \
--use_stopwords $use_sw --use_full_vocab $use_full_vocab --agg_by $agg_by

done
