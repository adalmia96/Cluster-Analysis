##!/usr/bin/env bash
# Author: Suzanna Sia
 
ALGO=SPKMeans
EMBEDS=/export/c12/ssia/shared/Cluster-Analysis/embeds

mkdir -p results/reuters
mkdir -p logs/qsub
mkdir -p logs/qsub_e

for ALGO in SPKMeans VMFM; do
  for emb in word2vec fasttext jose_300d.txt reuters-bert-layer12-average.full_vocab.fix reuters-elmo.full_vocab; do

    echo "running $emb... $ALGO"
    
    for weighted in 0 1; do
      for rr in 0 1; do
        qsub -N Cl.$weighted.$rr.$ALGO.$emb -o logs/qsub -e logs/qsub_e ./bin/quick_run.sh $EMBEDS $emb $ALGO $weighted $rr
        sleep 10
        #bash ./bin/quick_run.sh $EMBEDS $emb $ALGO $weighted $rr
      done
    done
  done
done

