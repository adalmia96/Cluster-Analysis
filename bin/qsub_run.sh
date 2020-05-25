##!/usr/bin/env bash
# Author: Suzanna Sia
 
EMBEDS=/export/c12/ssia/shared/Cluster-Analysis/embeds
DS=20NG

mkdir -p results/$DS
mkdir -p logs/qsub
mkdir -p logs/qsub_e

for ALGO in VMFM; do
  for emb in word2vec fasttext jose_300d.txt $DS-bert-layer12-average.full_vocab.fix $DS-elmo.full_vocab; do

    echo "running $emb... $ALGO"
    
    for weighted in 0 1; do
      for rr in 0 1; do
        qsub -N Cl.$weighted.$rr.$ALGO.$emb -o logs/qsub -e logs/qsub_e ./bin/quick_run.sh $DS $EMBEDS $emb $ALGO $weighted $rr
        sleep 10
        #bash ./bin/quick_run.sh $DS $EMBEDS $emb $ALGO $weighted $rr
      done
    done
  done
done

