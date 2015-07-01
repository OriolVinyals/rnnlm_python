#!/usr/bin/env bash
#===============================================================================
#
#          FILE: preprocess_data.sh
# 
#         USAGE: ./preprocess_data.sh 
# 
#   DESCRIPTION: Preprocess tinyshakespeare corpus
# 
#         NOTES: ---
#       CREATED: 06/20/15 16:49
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e

datadir=data/tinyshakespeare
infile=${datadir}/tinyshakespeare.txt
outfile=${datadir}/tinyshakespeare_char.txt

python src/util/preprocess_data.py \
  --infile ${infile} \
  --outfile ${outfile} \
  --lowercase \
  --empty-line
  #--clean-token \

# split data into train, valid and test
# train 1-2000 lines
# valid 2001-3000 lines
# the rest is kept as test set

trainfile=${datadir}/train.txt
vocfile=${datadir}/voc.txt
validfile=${datadir}/valid.txt
testfile=${datadir}/test.txt

train_lb=1
train_ub=2000
valid_lb=2001
valid_ub=3000
test_lb=3001

# split training set
awk -v lb=${train_lb} -v ub=${train_ub} \
  'NR>=lb && NR <=ub' \
  ${outfile} > ${trainfile} 

# split valid set
awk -v lb=${valid_lb} -v ub=${valid_ub} \
  'NR>=lb && NR <=ub' \
  ${outfile} > ${validfile} 

# split test set
awk -v lb=${test_lb} \
  'NR>=lb' \
  ${outfile} > ${testfile}

# build vocabulary
sed "s/ /\n/g" ${trainfile} | \
  sort | uniq > ${vocfile}
# append eos, unk, append tokens
echo "</s>" >> ${vocfile}
echo "<unk>" >> ${vocfile}
echo "<append>" >> ${vocfile}


