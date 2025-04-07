#!/bin/bash

# get directory with data
data_dir="${DATA_DIR:-data}"
mkdir $data_dir
cd $data_dir || exit

# Download data
wget http://download.cs.stanford.edu/deep/ForestNetDataset.zip
unzip ForestNetDataset.zip
mv deep/downloads/ForestNetDataset .

# Remove zip and empty tmp dir
rm ForestNetDataset.zip
rm -r deep


# (~3 Gb, ~40 min)