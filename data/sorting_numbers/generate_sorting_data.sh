#!/usr/bin/env bash
#===============================================================================
#
#          FILE: generate_sorting_data.sh
# 
#         USAGE: ./generate_sorting_data.sh 
# 
#   DESCRIPTION: 
# 
#         NOTES: ---
#       CREATED: 06/22/2015 15:32
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e

# train valid test filenames
tasks="train valid test"
trainfile=train
split_size=(1000 200 10)
vocabfile=vocab


seq_low=0
seq_high=10
seq_len=5



declare -i idx
idx=0
for task in ${tasks}; 
do
  echo "${task}"
  echo "${split_size[${idx}]}"
  size=${split_size[${idx}]}
  python ./generate_sorting_number.py \
    --low ${seq_low} \
    --high ${seq_high} \
    --size ${size} \
    --seq-len ${seq_len} \
    --outfn ${task}
  idx=idx+1
done

sed "s/ /\n/g" ${trainfile} | sort | \
  uniq > ${vocabfile}
echo "</s>" >> ${vocabfile}
echo "<unk>" >> ${vocabfile}

