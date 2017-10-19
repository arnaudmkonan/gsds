#!/bin/bash

# create list of correct labels that are available for training
# should run after test set is created!

touch data/corrected_labels_training.csv

for i in `ls data/retail`; do
    echo  'retail,' $i >> data/corrected_labels_training.csv
done

for i in `ls data/other`; do
    echo  'other,' $i >> data/corrected_labels_training.csv
done
