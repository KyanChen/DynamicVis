#!/bin/bash

## Run inference for ForestNet
## python inference.py -c configs/prithvi_vit.yaml -d ForestNet -s train
# python inference.py -c configs/prithvi_vit.yaml -d ForestNet -s val
# python inference.py -c configs/prithvi_vit.yaml -d ForestNet -s test
#
## python inference.py -c configs/prithvi_vit.yaml -d ForestNet4 -s train
# python inference.py -c configs/prithvi_vit.yaml -d ForestNet4 -s val
# python inference.py -c configs/prithvi_vit.yaml -d ForestNet4 -s test
#
## python inference.py -c configs/prithvi_vit.yaml -d ForestNetBGR -s train
# python inference.py -c configs/prithvi_vit.yaml -d ForestNetBGR -s val
# python inference.py -c configs/prithvi_vit.yaml -d ForestNetBGR -s test
#
## python inference.py -c configs/prithvi_vit.yaml -d ForestNet4BGR -s train
# python inference.py -c configs/prithvi_vit.yaml -d ForestNet4BGR -s val
# python inference.py -c configs/prithvi_vit.yaml -d ForestNet4BGR -s test
#
## Run inference for BigEarthNet
##python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet -s train
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet -s val
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet -s test
#
##python inference.py -c configs/prithvi_vit.yaml -d BigEarthNetBGR -s train
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNetBGR -s val
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNetBGR -s test
#
##python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19 -s train
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19 -s val
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19 -s test
#
##python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19BGR -s train
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19BGR -s val
#python inference.py -c configs/prithvi_vit.yaml -d BigEarthNet19BGR -s test
#
## Run inference with vanilla ViT
## python inference.py -c configs/rgb_vit.yaml -d ForestNetBGR -s train
# python inference.py -c configs/rgb_vit.yaml -d ForestNetBGR -s val
# python inference.py -c configs/rgb_vit.yaml -d ForestNetBGR -s test
#
## python inference.py -c configs/rgb_vit.yaml -d ForestNet4BGR -s train
# python inference.py -c configs/rgb_vit.yaml -d ForestNet4BGR -s val
# python inference.py -c configs/rgb_vit.yaml -d ForestNet4BGR -s test
#
##python inference.py -c configs/rgb_vit.yaml -d BigEarthNetBGR -s train
#python inference.py -c configs/rgb_vit.yaml -d BigEarthNetBGR -s val
#python inference.py -c configs/rgb_vit.yaml -d BigEarthNetBGR -s test
#
##python inference.py -c configs/rgb_vit.yaml -d BigEarthNet19BGR -s train
#python inference.py -c configs/rgb_vit.yaml -d BigEarthNet19BGR -s val
#python inference.py -c configs/rgb_vit.yaml -d BigEarthNet19BGR -s test

#pip install torchgeo h5py lshashpy3
# Run inference with vanilla ViT
#python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d ForestNetRGB -s val --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d ForestNetRGB -s test --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d ForestNet4RGB -s val --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d ForestNet4RGB -s test --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/
#
#CUDA_VISIBLE_DEVICES=0 python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d BigEarthNetRGB -s val --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/
#CUDA_VISIBLE_DEVICES=1 python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d BigEarthNetRGB -s test --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/
#CUDA_VISIBLE_DEVICES=2 python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d BigEarthNet19RGB -s val --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/
#CUDA_VISIBLE_DEVICES=3 python retrieval/inference.py -c retrieval/configs/rgb_vit.yaml -d BigEarthNet19RGB -s test --input_size 224 --data_dir /mnt/nlp-ali/dataset/cky_data/


#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d ForestNetRGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d ForestNetRGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d ForestNet4RGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d ForestNet4RGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#
#CUDA_VISIBLE_DEVICES=0 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d BigEarthNetRGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#CUDA_VISIBLE_DEVICES=1 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d BigEarthNetRGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#CUDA_VISIBLE_DEVICES=2 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d BigEarthNet19RGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#CUDA_VISIBLE_DEVICES=3 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis.yaml --arch b -d BigEarthNet19RGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/


#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d ForestNetRGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d ForestNetRGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d ForestNet4RGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d ForestNet4RGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#
#CUDA_VISIBLE_DEVICES=0 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d BigEarthNetRGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/ &
#CUDA_VISIBLE_DEVICES=1 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d BigEarthNetRGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/ &
#CUDA_VISIBLE_DEVICES=2 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d BigEarthNet19RGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/ &
#CUDA_VISIBLE_DEVICES=3 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_l.yaml --arch l -d BigEarthNet19RGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/


#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d ForestNetRGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d ForestNetRGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d ForestNet4RGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
#python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d ForestNet4RGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/

CUDA_VISIBLE_DEVICES=0 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d BigEarthNetRGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/ &
CUDA_VISIBLE_DEVICES=1 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d BigEarthNetRGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/ &
CUDA_VISIBLE_DEVICES=2 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d BigEarthNet19RGB -s val --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/ &
CUDA_VISIBLE_DEVICES=3 python retrieval/inference.py -c retrieval/configs/rgb_dynamicvis_roi_l.yaml --arch l -d BigEarthNet19RGB -s test --input_size 512 --data_dir /mnt/nlp-ali/dataset/cky_data/
