##!/usr/bin/env bash
 
EMBEDS=/export/c12/ssia/shared/Cluster-Analysis/embeds
DS=20NG

mkdir -p results/$DS
mkdir -p logs/qsub
mkdir -p logs/qsub_e

for ALGO in SPKMeans; do # GMM SPKMeans; do #VMFM; do
  #for emb in glove word2vec fasttext jose_300d.txt $DS-bert-layer12-average.full_vocab.fix $DS-elmo.full_vocab.layer2; do
  for emb in word2vec; do #word2vec; do #$DS-elmo.full_vocab.layer2; do #jose_300d.txt $DS-bert-layer12-average.full_vocab.fix; do 
    echo "running $emb... $ALGO"
  #  for scale in log; do 
    for weighted in 0 1; do
      for rr in 0 1; do
        qsub -N Cl.$weighted.$rr.$ALGO.$emb -o logs/qsub -e logs/qsub_e ./bin/quick_run.sh $DS $EMBEDS $emb $ALGO $weighted $rr
        sleep 10
        #bash ./bin/quick_run.sh $DS $EMBEDS $emb $ALGO $weighted $rr $scale
      done
    done
   # done
  done
done

