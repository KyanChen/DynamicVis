#!/bin/bash

# get directory with data
data_dir="${DATA_DIR:-data}"
mkdir $data_dir
cd $data_dir || exit

# create BigEarthNet dir
mkdir BigEarthNet
cd BigEarthNet || exit

# download splits
wget https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/train.csv -O bigearthnet-train.csv
wget https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/val.csv -O bigearthnet-val.csv
wget https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/test.csv -O bigearthnet-test.csv

# download BigEarthNet
curl -O https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz
tar -xvzf BigEarthNet-S2-v1.0.tar.gz
# delete tar file
rm BigEarthNet-S2-v1.0.tar.gz
