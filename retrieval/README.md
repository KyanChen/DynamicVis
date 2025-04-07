## Datasets

We provide bash script for downloading the datasets. Run the following script from the project root:

```shell
# optionally specific the data directory (default: 'data/')
export DATA_DIR=data

# Download BigEarthNet (~66 Gb, ~1h)
sh datasets/bigearthnet_download.sh

# Download ForestNet (3 Gb, ~5 min)
sh datasets/forestnet_download.sh
```

## Models

You can download the model weights for Prithvi-100M from [Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M.pt) with the following commands.

```shell
mkdir weights
cd weights && wget https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt
```

The weights are saved at `weights/Prithvi_100M.pt` but you can also update the path in the config file `configs/prithvi_vit_us.yaml`.

The weights for the vanilla ViT with RGB channels are downloaded automatically. 


## Run experiments

You can save the embeddings of a dataset with:
```shell
# Save embeddings
python inference.py -c configs/prithvi_vit.yaml --dataset ForestNet --split val
python inference.py -c configs/prithvi_vit.yaml --dataset ForestNet --split test
```

If you want to save the embeddings of all evaluated models and dataset versions, you can run:
```shell
bash inference.sh
```

Evaluate all saved embeddings with given 
```shell
# Run experiments
python experiments.py --match any --distance_function hamming --hash_method trivial --hash_length 32
# You can also combine multiple methods
python experiments.py --match any --distance_function hamming --hash_method trivial,lsh,none --hash_length 32,768
```


## Speed experiments

You need a running [Milvus](https://milvus.io) instance for these experiments.

With saved BigEarthNet embeddings, run the experiments with:
```shell
python speed_test_milvus.py
```

If you want to run the experiments on another machine, connect to Milvus via ssh. 

```shell
ssh <server> -L19530:localhost:19530
```

## Thanks

We are grateful to the [Remote sensing image retrieval](https://github.com/IBM/remote-sensing-image-retrieval) project.
