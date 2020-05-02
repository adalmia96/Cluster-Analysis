#!/usr/bin/env bash
#Author: Suzanna Sia

TrainTextDir=/export/c12/ssia/shared/Cluster-Analysis/data/for_lda
MALLETDIR=/home/ssia/packages/Mallet

# Modify This
SAVEDIR=/home/ssia/projects/Cluster-Analysis/mallet_temp
NTOPICS=20

mkdir -p $SAVEDIR/

TransF=$SAVEDIR/topic-input.Mallet
Inferencer=$SAVEDIR/topic-inferencer
TopicModel=$SAVEDIR/topic-model
TransTopics=$SAVEDIR/TargetTopics.txt
TopicWords=$SAVEDIR/TopicWords.txt

printf "Format Train Docs to mallet..\n"
$MALLETDIR/bin/mallet import-dir --input $TrainTextDir --output $TransF --keep-sequence --remove-stopwords

printf "Running Topic Model..\n"
$MALLETDIR/bin/mallet train-topics --input $TransF --num-topics $NTOPICS --output-doc-topics \
$TransTopics --output-state $SAVEDIR/topic-state.gz --output-model $TopicModel \
--output-topic-keys $TopicWords 

#printf "Convert model to inferencer..\n"
#$MALLETDIR/bin/mallet train-topics --input-model $TopicModel --inferencer-filename $Inferencer --num-iterations 0

#printf "Infer topics from new docs..\n"
#$MALLETDIR/bin/mallet infer-topics --inferencer $Inferencer --input $TransF --output-doc-topics $QueryTopicsTrain

#printf "Infer top words from topic..\n"
#this doesnt work
#$MALLETDIR/bin/mallet infer-topics --inferencer $Inferencer --input $TransF --output-topic-keys $TopicWords


printf "done"
