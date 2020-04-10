#!/usr/bin/env bash

for x in {8..12}; do
  qsub -N layer$x -o logs/qsub -e logs/qsub_e ./bin/run2.sh $x
done
