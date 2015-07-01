#!/usr/bin/env bash
#===============================================================================
#
#          FILE: sample_rnn_lm.sh
# 
#         USAGE: ./sample_rnn_lm.sh 
# 
#   DESCRIPTION: 
# 
#         NOTES: ---
#        AUTHOR: Hao Cheng, chenghao@uw.edu
#       CREATED: 06/29/2015 22:05
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e


vocabfile=data/tinyshakespeare/voc.txt
inmodel=expts/debug.char-lm.20150620

python ./src/sample_rnn_lm.py \
  --inmodel ${inmodel} \
  --vocabfile ${vocabfile} \
  --spell-word \
  --sample-sent
