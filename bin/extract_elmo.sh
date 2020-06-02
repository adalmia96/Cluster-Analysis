#!/usr/bin/env bash
#Author: Suzanna Sia
#$ -l 'hostname=c*,mem_free=10G,ram_free=10G,gpu=1'
#$ -cwd
#$ -q g.q
#$ -m ea
#$ -M fengsf85@gmail.com

#conda activate allennlp
source activate allennlp

CUDA=`free-gpu`
export CUDA_VISIBLE_DEVICES=$CUDA
CUDA=0
SAVEDIR=/export/c12/ssia/shared/Cluster-Analysis/embeds
DATA=20NG # 20NG or cb
use_full_vocab=1
use_sw=0

#for DATA in 20NG cb reuters; do
#for DATA in fetch20 children reuters; do
for DATA in reuters; do

  SAVEFN=$SAVEDIR/${DATA}-elmo.full_vocab.layer2
  python code/elmo_encode.py --device $CUDA --data $DATA --save_fn $SAVEFN --use_stopwords \
  $use_sw --use_full_vocab $use_full_vocab --mixture_coefficient "0;0;1"

done
