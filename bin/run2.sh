#!/usr/bin/env bash
#$ -l 'hostname=b1[12345678]*|c*,mem_free=5G,ram_free=5G,gpu=1'
#$ -cwd
#$ -q g.q
#$ -m ea
#$ -M fengsf85@gmail.com

CUDA=`free-gpu`
export CUDA_VISIBLE_DEVICES=$CUDA

python bert_encode.py --layer $1
