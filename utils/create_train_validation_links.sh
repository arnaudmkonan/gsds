#!/bin/bash

# note run test set creation before this script

# to run for 80:20 split
# bash utils/create_train_validation.sh

if [ "$#" != "1" ]; then
  echo "Error: supply validation split factor, eg 5 for an 80:20 split, or 3 for a one-third split"
  exit 0
fi

split_factor=$1

## create training and validation and test sets

# if these exist we want to remove them
[ -d "data/train$split_factor/retail" ] && rm -rf data/train$split_factor/retail
[ -d "data/train$split_factor/other" ] && rm -rf data/train$split_factor/other
[ -d "data/val$split_factor/retail" ] && rm -rf data/val$split_factor/retail
[ -d "data/val$split_factor/other" ] && rm -rf data/val$split_factor/other

mkdir -p data/train$split_factor/retail
mkdir -p data/train$split_factor/other
mkdir -p data/val$split_factor/retail
mkdir -p data/val$split_factor/other

## stratified split based on randomized 80:20 holdout for each class

total_retail=`ls data/retail | wc -l`
echo Total retail/restaurant images: $total_retail
let "rv = $total_retail / $split_factor"
echo Validation total based on 1 / $split_factor : $rv

# shuf is a linux tool, on mac may need to install or use gshuf
ls data/retail/* | shuf > aaa
tval=`cat aaa | head -n $rv`
let "rt=rv+1"
ttrain=`cat aaa | tail -n +$rt`
rm aaa

cd data/val$split_factor/retail/
for i in $tval; do
  ln -s ../../../$i .
done
cd ../../..

cd data/train$split_factor/retail/
for i in $ttrain; do
  ln -s ../../../$i .
done
cd ../../..


total_other=`ls data/other | wc -l`
echo Total other images: $total_other
let "ov = $total_other / $split_factor"
echo Validation total based on 1 / $split_factor : $ov

ls data/other/* | shuf > aaa
toval=`cat aaa | head -n $ov`
let "ot=ov+1"
otrain=`cat aaa | tail -n +$ot`
rm aaa

cd data/val$split_factor/other/
for i in $toval; do
  ln -s ../../../$i .
done
cd ../../..

cd data/train$split_factor/other/
for i in $otrain; do
  ln -s ../../../$i .
done
cd ../../..

# check directories have correct amount of files
echo "Retail validation count:" `ls data/val$split_factor/retail | wc -l`
echo "Other  validation count:" `ls data/val$split_factor/other | wc -l`
