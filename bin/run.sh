#!/usr/bin/env bash
#Author: Suzanna Sia
for Algo in KMeans GMM
  do
  for dim in 150 #10 50 100 200 300
    do
      echo "Running $Algo with dim $dim";
      python score.py --entities fasttext --clustering_algo $Algo --use_dims $dim
  done
done

