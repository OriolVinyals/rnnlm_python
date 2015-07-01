#!/usr/bin/env bash
#===============================================================================
#
#          FILE: train_rnn_sort.sh
# 
#         USAGE: ./train_rnn_sort.sh 
# 
#   DESCRIPTION: Train a RNN sorting model
# 
#         NOTES: ---
#       CREATED: 06/20/15 19:19
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e

datadir=data/sorting_numbers
trainfile=${datadir}/train
validfile=${datadir}/valid
vocabfile=${datadir}/vocab
outmodel=expts/debug.sort.20150620
separator=\<sort\>
nhidden=100
initalpha=1e-2
batchsize=1
tol=0.5
bptt=12

python ./src/train_rnn_lm.py \
  --trainfile ${trainfile} \
  --validfile ${validfile} \
  --vocabfile ${vocabfile} \
  --separator ${separator} \
  --init-alpha ${initalpha} \
  --batchsize ${batchsize} \
  --nhidden ${nhidden} \
  --outmodel ${outmodel} \
  --tol ${tol} \
  --bptt ${bptt} \
  --validate

