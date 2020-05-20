#!/usr/bin/env bash
#Author: Suzanna Sia

#for x in firstword average; do
#  qsub -N Bl12 -o logs/qsub -e logs/qsub_e ./bin/run2.sh $x
#done

qsub -N elmo -o logs/qsub -e logs/qsub_e ./bin/runelmo.sh $x
#qsub -N Bl12 -o logs/qsub -e logs/qsub_e ./bin/run2.sh average 

