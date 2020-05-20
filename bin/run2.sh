#!/usr/bin/env bash
# Author:Suzanna Sia
#$ -l 'hostname=c*,mem_free=5G,ram_free=5G,gpu=1'
#$ -cwd
#$ -q g.q
#$ -m ea
#$ -M fengsf85@gmail.com

CUDA=`free-gpu`
export CUDA_VISIBLE_DEVICES=$CUDA
CUDA=0

python code/bert_encode.py --nlayer 12 --device $CUDA --agg_by $1 --save_fn embeds/20NG-bert-layer12-$1.txt.swr --data 20NG
