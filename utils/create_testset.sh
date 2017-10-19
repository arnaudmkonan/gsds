#!/bin/bash

# to create test set of 100 files (remove 50 from retail and 50 from other):
#    bash utils/create_testset.sh 50

# first we split the data into two folders: one for other and one for retail and/or restaurant
# run to move images from all_images directory to other and retail

# needed for gnu tools, eg grep to work if want to work with these labels
# dos2unix data/practical_stuff/labels.txt

if [ "$#" != "1" ]; then
  echo "Error: supply amount of images to remove from each class for test set"
  exit 0
fi

sample=$1

[ -d "data/practical_stuff" ] && rm -rf data/practical_stuff
cd data
unzip practical_stuff.zip
cd ..

# remove directory if it exists
[ -d "data/retail" ] && rm -rf data/retail
mkdir data/retail
# note there are some different spellings of restaurant in original labels so we look for retail only
for i in `cat data/corrected_labels.csv | grep retail | awk '{print $2}'`; do
    mv data/practical_stuff/all_images/$i data/retail/$i
done

[ -d "data/other" ] && rm -rf data/other
mkdir data/other
for i in `cat data/corrected_labels.csv | grep other | awk '{print $2}'`; do
   mv data/practical_stuff/all_images/$i data/other/$i
done

echo "Done splitting dataset!"

# create test set by moving files from our data
[ -d "data/test/retail" ] && rm -rf data/test/retail
mkdir -p data/test/retail
[ -d "data/test/other" ]  && rm -rf data/test/other
mkdir -p data/test/other

echo "Removing $sample files from each class for test holdout set"
# create retail test set
for i in `ls data/retail/* | shuf | head -n $sample`; do
    mv $i data/test/retail/
done
echo total files in retail test `ls data/test/retail/ | wc -l`

# create other test set
for i in `ls data/other/* | shuf | head -n $sample`; do
    mv $i data/test/other/
done
echo total files in other test `ls data/test/other/ | wc -l`

echo "Done creating test set!"
