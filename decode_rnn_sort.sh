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


vocabfile=data/sorting_numbers/vocab
inmodel=expts/debug.sort.20150620

python ./src/decode_rnn_sort.py \
  --inmodel ${inmodel} \
  --vocabfile ${vocabfile}
