#!/usr/bin/env bash
#$ -l 'hostname=c*,mem_free=5G,ram_free=5G,gpu=1'
#$ -cwd
#$ -q g.q
#$ -m ea
#$ -M fengsf85@gmail.com

conda activate allennlp

CUDA=`free-gpu`
export CUDA_VISIBLE_DEVICES=$CUDA

python code/elmo_encode.py --nlayer 12 --device $CUDA --save_fn embeds/20ng-elmo-layer3.txt.swr
