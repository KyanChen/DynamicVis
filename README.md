<div align="center">
    <h2>
       DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/DynamicVis.png" width="800"/>
</div>
<br>
<div align="center">
  <a href="https://github.com/KyanChen/DynamicVis">
    <span style="font-size: 20px; ">Homepage</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2503.16426">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="resources/DynamicVis.pdf">
    <span style="font-size: 20px; ">PDF</span>
  </a>
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/KyanChen/DynamicVis)](https://github.com/KyanChen/DynamicVis)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2503.16426-b31b1b.svg)](https://arxiv.org/abs/2503.16426)

<br>
<br>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>


## Introduction

This repository contains the official implementation of the paper [DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding](https://arxiv.org/abs/2503.16426), developed based on the [OpenMMLab](https://openmmlab.com/codebase) framework.

DynamicVis is a dynamic visual perception foundation model for remote sensing, achieving efficient low-resource parsing of ultra-large images (2048x2048 pixels processing requires only 800MB GPU RAM) through a selective region-aware architecture and multi-instance meta-embedding learning. The model demonstrates exceptional performance across nine remote sensing downstream tasks, with \~20x computational efficiency and \~97% memory reduction compared to ViT, enabling cross-task understanding of high-resolution remote sensing imagery.

The current branch has been tested on Linux systems with PyTorch 2.x and CUDA 12.1, supporting Python 3.10+ and compatible with most CUDA versions.

If you find this project helpful, please give us a star ⭐️. Your support is our greatest motivation.



<details open>
<summary>Main Features</summary>

- API interfaces and usage methods highly consistent with OpenMMLab
- Open-sourced DynamicVis models and weights of different scales as described in the paper
- Supports fine-tuning and testing for nine remote sensing downstream tasks mentioned in the paper

</details>

## Changelog

🌟 **2025.03.20** Released DynamicVis project.

🌟 **2025.03.21** Updated DynamicVis pretraining code.

🌟 **2025.03.22** Updated the scene classification fine-tuning code for DynamicVis.

🌟 **2025.03.22** Updated the tiny object detection fine-tuning code for DynamicVis.

🌟 **2025.03.31** Updated the fine-tuning code for all tasks of DynamicVis.

🌟 **2025.04.01** Uploaded the pretraining weights of DynamicVis.


## TODO

- [X] Organize DynamicVis pretraining code
- [X] Organize fine-tuning and testing code for nine tasks in the paper
- [X] Upload DynamicVis model weights
- [X] Upload DynamicVis model weights without token selection


## Contents

- [Introduction](#introduction)
- [Changelog](#changelog)
- [TODO](#todo)
- [Table of Contents](#Contents)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Pretraining](#model-pretraining)
- [Model Fine-tuning](#model-fine-tuning)
- [Pretrained Weights Download](#pretrained-weights-download)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Linux OS (Windows not supported for Mamba)
- Python 3.10+ (3.11 recommended)
- PyTorch 2.0+ (2.4 recommended)
- CUDA 11.7+ (12.1 recommended)
- MMCV 2.0+ (2.2 recommended)
- Mamba 2.2.4

### Environment Setup

We recommend using Miniconda for installation. The following commands will create a virtual environment named `dynamicvis` and install PyTorch and MMCV. The default CUDA version in these instructions is **12.1**. Modify accordingly if using a different CUDA version.

<details>

**Step 0**: Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

**Step 1**: Create and activate a virtual environment:

```shell
conda create -n dynamicvis python=3.11 -y
conda activate dynamicvis
```

**Step 2**: Install [PyTorch 2.4.x](https://pytorch.org/get-started/previous-versions/).

Linux:

```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# OR
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV 2.2.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install mmcv==2.2.0
# OR
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

**Step 4**: Install [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) and [Mamba 2.2.4](https://github.com/state-spaces/mamba):

```shell
# Install causal-conv1d
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Install mamba
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.4+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

Note: If you encounter issues, try using `pip install`.

```shell
pip install causal-conv1d==1.5.0.post8
pip install mamba==2.2.4

# OR
# Compile and install first, and keep the whl file for future installations to avoid recompilation
# pip wheel --wheel-dir=../../software/mamba2-2.4.4/ causal-conv1d==1.5.0.post8 -i https://pypi.org/simple
# pip wheel --wheel-dir=../../software/mamba2-2.4.4/ mamba-ssm==2.2.4 -i https://pypi.org/simple
# pip install ../../software/mamba2-2.4.4/causal_conv1d-1.5.0.post8-cp311-cp311-linux_x86_64.whl
# pip install ../../software/mamba2-2.4.4/mamba_ssm-2.2.4-cp311-cp311-linux_x86_64.whl
```



**Step 5**: Install other dependencies:

```shell
pip install transformers==4.50.0 
pip install -U ipdb braceexpand mat4py pycocotools shapely ftfy scipy terminaltables wandb prettytable torchmetrics importlib_metadata einops
```

</details>


### Install DynamicVis

Download or clone the DynamicVis repository。

```shell
git clone git@github.com:KyanChen/DynamicVis.git
cd DynamicVis
```

## Dataset Preparation


### Pretraining Dataset


<details>

##### Download Data

- Dataset: [fMoW Dataset](https://github.com/fMoW/dataset)
- Download the **fMoW-rgb** subset:

```shell
pip install awscli

# Delete AWS configuration file
rm -rf ~/.aws
# List files
aws s3 ls --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/

# Download files
aws s3 sync --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/ ./data/fmow-rgb/
```

##### Organization

We use [WebDataset](https://github.com/webdataset/webdataset) to organize large-scale pretraining data. WebDataset is a data loading library for large-scale datasets that can efficiently handle large datasets.

```shell
# Reorganize the data into Tar package format required by WebDataset
python tools_DynamicVis/tools_data/fMoW/get_fmow_train_val_data.py
```

</details>


### Scene Classification Datasets

<details>

##### Download Data

- [UC Merced Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)。
- [AID Dataset](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)。

##### Organization

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/data/UC
├── airplane
│   ├── airplane01.tif
│   ├── airplane02.tif
│   └── ...
├── ...
├── ...
├── ...
└── ...
```

**Note**: In the project folder `datainfo`, we provide the split files for the datasets. You can also use the [Python script](tools_DynamicVis/tools_data/UCMerced/split_trainval.py) to split the datasets.


</details>



### Tiny Object Detection Datasets

<details>

#### Download Data

- [Levir-Ship Dataset](https://github.com/WindVChen/LEVIR-Ship?tab=readme-ov-file#Download-Source), download the COCO format dataset annotations.

#### Dataset Organization

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/data/Levir-Ship
├── images
│   ├── train
│   │   ├── train_0.jpg
│   │   ├── ...
│   │   └── ...
│   └── val
│   │   ├── val_0.jpg
│   │   ├── ...
│   │   └── ...
│   └── test
│   │   ├── test_0.jpg
│   │   ├── ...
│   │   └── ...
├── annotations
│   ├── train.json
│   ├── val.json
│   └── test.json
 ```

</details>

### Instance Segmentation Datasets

<details>

#### Download Data

- [NWPU VHR-10 Image Data](https://aistudio.baidu.com/datasetdetail/52812).
- [NWPU VHR-10 Instance Annotations](https://github.com/chaozhong2010/VHR-10_dataset_coco).



- [SSDD Image Data](https://aistudio.baidu.com/datasetdetail/56503).
- [SSDD Instance Annotations](https://github.com/chaozhong2010/VHR-10_dataset_coco).


**Note**: In the `datainfo` folder of this project, we provide the instance annotations for the above datasets, which you can use directly.

#### Dataset Organization

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/data/NWPU
├── images
│   ├── train
│   │   ├── train_0.jpg
│   │   ├── ...
│   │   └── ...
│   └── val
│   │   ├── val_0.jpg
│   │   ├── ...
│   │   └── ...
│   └── test
│   │   ├── test_0.jpg
│   │   ├── ...
│   │   └── ...
├── annotations
│   ├── train.json
│   ├── val.json
│   └── test.json
```

</details>


### Semantic Segmentation Datasets


<details>

#### Download Data

- [Massachusetts Roads Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset/data).
- [WHU Building Dataset](https://aistudio.baidu.com/datasetdetail/56502).

#### Dataset Organization

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/Massachusetts
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/data/Massachusetts
├── train_imgs
│   ├── train_0.tif
│   ├── ...
│   └── ...
├── val_imgs
│   ├── val_0.tif
│   ├── ...
│   └── ...
├── test_imgs
│   ├── test_0.tif
│   ├── ...
│   └── ...
├── train_labels
│   ├── train_0.tif
│   ├── ...
│   └── ...
├── val_labels
│   ├── val_0.tif
│   ├── ...
│   └── ...
├── test_labels
│   ├── test_0.tif
│   ├── ...
│   └── ...
```
</details>


### Change Detection Datasets


<details>

#### Download Data

- [LEVIR-CD Dataset](https://aistudio.baidu.com/datasetdetail/202366).
- [WHU-CD Dataset](https://aistudio.baidu.com/datasetdetail/251669).
- [OSCD Dataset](https://rcdaudt.github.io/oscd/).

#### Dataset Organization

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/LEVIR-CD
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/data/LEVIR-CD
├── A
│   ├── A_0.jpg
│   ├── ...
│   └── ...
├── B
│   ├── B_0.jpg
│   ├── ...
│   └── ...
├── labels
│   ├── A_0.png
│   ├── ...
│   └── ...
```

**Note**: We provide the relevant processing code for the data in the [tools_DynamicVis/tools_data](tools_DynamicVis/tools_data) folder.

</details>





## Model Pretraining

### Config File Overview

We provide configuration files for DynamicVis models of different parameter sizes as described in the paper. You can find them in the [configs_DynamicVis/fMoW](configs_DynamicVis/fMoW) folder. The config files maintain consistent API interfaces and usage methods with OpenMMLab. Below are some key parameter explanations. For more information on the parameters, refer to the [OpenMMLab documentation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/1_config.html).


<details>

**Parameter Explanation**:

- `work_dir`: Output path for model training, generally no need to modify.
- `data_root`: Dataset root directory, **modify to the absolute path of the dataset root**.
- `code_root`: Code root directory, **modify to the absolute path of the code root**.
- `batch_size`: Batch size per GPU, **modify according to GPU memory size**.
- `max_epochs`: Maximum number of training epochs, generally no need to modify.
- `val_interval`: Interval of validation set, generally no need to modify.
- `vis_backends/WandbVisBackend`: Configuration of network-side visualization tools, **after uncommenting, you need to register an account on the `wandb` official website to view the visualization results during training in a web browser**.
- `load_from`: Path to the model's pretraining checkpoint, generally no need to modify.
- `resume`: Whether to resume training from a checkpoint, generally no need to modify.
- `default_hooks/CheckpointHook`: Configuration of model checkpoint saving during training, generally no need to modify.
- `model/backbone`: Visual backbone of the DynamicVis model, **modify according to actual situation**.
- `model/backbone/arch`: Configuration of the main network, **modify according to actual situation**.
- `model/backbone/spatial_token_keep_ratios`: Spatial token retention ratio, **modify according to actual situation**.
- `model/pre_neck`: FPN Neck of the DynamicVis model.
- `model/neck`: Region feature extractor of the DynamicVis model, generally no need to modify.
- `model/head`: Classification head of the DynamicVis model, generally no need to modify.
- `optim_wrapper`: Configuration of the optimizer, generally no need to modify.
- `data_preprocessor/mean/std`: Mean and standard deviation of data preprocessing, generally no need to modify.



### Training Commands

```shell
# Single GPU
python tools_mmpretrain/train.py configs_DynamicVis/fMoW/name_to_config.py
# Multi-GPU
sh tools_mmpretrain/dist_train.sh configs_DynamicVis/fMoW/name_to_config.py ${GPU_NUM}
```

### Testing

```shell
# Single GPU
python tools_mmpretrain/test.py configs_DynamicVis/fMoW/name_to_config.py ${CHECKPOINT_FILE}
# Multi-GPU
sh tools_mmpretrain/dist_test.sh configs_DynamicVis/fMoW/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}
```
</details>


## Model Fine-tuning

### Scene Classification

<details>

#### Config Files

We provide configuration files for the UC Merced and AID datasets mentioned in the paper. You can find them in the [UC configuration file](configs_DynamicVis/UCMerced) and [AID configuration file](configs_DynamicVis/AID) folders.


The following are some key parameter explanations other than the pretraining part of the Config.


**Parameter Explanation**:

- `pretrained_ckpt`：模型微调的预训练检查点路径，**需要根据实际情况进行修改**。
- `pretrained_ckpt`: Path to the pretrained checkpoint for model fine-tuning, **needs to be modified according to the actual situation**.


### Fine-tuning

```shell
# Single GPU
python tools_mmpretrain/train.py configs_DynamicVis/UCMerced/name_to_config.py  # name_to_config.py is the config file you want to use
# Multi-GPU
sh tools_mmpretrain/dist_train.sh configs_DynamicVis/UCMerced/name_to_config.py ${GPU_NUM}  # name_to_config.py is the config file you want to use, GPU_NUM is the number of GPUs used
```

### Testing

```shell
# Single GPU
python tools_mmpretrain/test.py configs_DynamicVis/UCMerced/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
# Multi-GPU
sh tools_mmpretrain/dist_test.sh configs_DynamicVis/UCMerced/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

</details>


### Tiny Object Detection

<details>

#### Config Files and Main Parameter Explanation

We provide configuration files for the Levir-Ship dataset mentioned in the paper. You can find them in the [Levir-Ship configuration file](configs_DynamicVis/Levir-Ship) folder.

The following are some key parameter explanations other than the pretraining part of the Config.

**Parameter Explanation**：

- `pretrained_ckpt`: Path to the pretrained checkpoint for model fine-tuning, **needs to be modified according to the actual situation**.
- `default_hooks/visualization`: Control whether to visualize during val and test, **need to modify the `draw` and `interval` parameters according to the actual situation**.

#### Fine-tuning

```shell
# Single GPU
python tools_mmdet/train.py configs_DynamicVis/Levir-Ship/name_to_config.py  # name_to_config.py is the config file you want to use
# Multi-GPU
sh tools_mmdet/dist_train.sh configs_DynamicVis/Levir-Ship/name_to_config.py ${GPU_NUM}  # name_to_config.py is the config file you want to use, GPU_NUM is the number of GPUs used
```

#### Testing

```shell
# Single GPU
python tools_mmdet/test.py configs_DynamicVis/Levir-Ship/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
# Multi-GPU
sh tools_mmdet/dist_test.sh configs_DynamicVis/Levir-Ship/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

</details>


### Instance Segmentation

<details>

#### Config Files and Main Parameter Explanation

We provide configuration files for the NWPU and SSDD datasets mentioned in the paper. You can find them in the [NWPU configuration file](configs_DynamicVis/NWPU) and [SSDD configuration file](configs_DynamicVis/SSDD) folders.


The following are some key parameter explanations other than the pretraining part of the Config.

**Parameter Explanation**：

- `pretrained_ckpt`: Path to the pretrained checkpoint for model fine-tuning, **needs to be modified according to the actual situation**.
- `default_hooks/visualization`: Control whether to visualize during val and test, **need to modify the `draw` and `interval` parameters according to the actual situation**.
- `visualizer`: Control the parameters during visualization, such as `line_width`, `alpha`, etc., **need to modify according to the actual situation**.

#### Fine-tuning

```shell
# Single GPU
python tools_mmdet/train.py configs_DynamicVis/NWPU/name_to_config.py  # name_to_config.py is the config file you want to use
# Multi-GPU
sh tools_mmdet/dist_train.sh configs_DynamicVis/NWPU/name_to_config.py ${GPU_NUM}  # name_to_config.py is the config file you want to use, GPU_NUM is the number of GPUs used
```

#### Testing

```shell
# Single GPU
python tools_mmdet/test.py configs_DynamicVis/NWPU/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py  is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
# Multi-GPU
sh tools_mmdet/dist_test.sh configs_DynamicVis/NWPU/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

</details>


### Semantic Segmentation

<details>

#### Config Files and Main Parameter Explanation

We provide configuration files for the Massachusetts and WHU datasets mentioned in the paper. You can find them in the [Massachusetts configuration file](configs_DynamicVis/Massachusetts) and [WHU configuration file](configs_DynamicVis/WHU) folders.

The following are some key parameter explanations other than the pretraining part of the Config.

**Parameter Explanation**：

- `pretrained_ckpt`: Path to the pretrained checkpoint for model fine-tuning, **needs to be modified according to the actual situation**.
- `default_hooks/visualization`: Control whether to visualize during val and test, **need to modify the `draw` and `interval` parameters according to the actual situation**.
- `visualizer`: Control the parameters during visualization, such as `alpha`, etc., **need to modify according to the actual situation**.

#### Fine-tuning

```shell
# Single GPU
python tools_mmseg/train.py configs_DynamicVis/Massachusetts/name_to_config.py  # name_to_config.py is the config file you want to use
# Multi-GPU
sh tools_mmseg/dist_train.sh configs_DynamicVis/Massachusetts/name_to_config.py ${GPU_NUM}  # name_to_config.py is the config file you want to use, GPU_NUM is the number of GPUs used
```

#### Testing

```shell
# Single GPU
python tools_mmseg/test.py configs_DynamicVis/Massachusetts/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
# Multi-GPU
sh tools_mmseg/dist_test.sh configs_DynamicVis/Massachusetts/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

</details>


### Change Detection

<details>

#### Config Files and Main Parameter Explanation

We provide configuration files for the LEVIR-CD, WHU-CD, and OSCD datasets mentioned in the paper. You can find them in the [LEVIR-CD configuration file](configs_DynamicVis/LEVIR-CD), [WHU-CD configuration file](configs_DynamicVis/WHU-CD), and [OSCD configuration file](configs_DynamicVis/OSCD) folders.

The following are some key parameter explanations other than the pretraining part of the Config.

**Parameter Explanation**：

- `pretrained_ckpt`: Path to the pretrained checkpoint for model fine-tuning, **needs to be modified according to the actual situation**.
- `default_hooks/visualization`: Control whether to visualize during val and test, **need to modify the `draw` and `interval` parameters according to the actual situation**.
- `visualizer`: Control the parameters during visualization, such as `alpha`, etc., **need to modify according to the actual situation**.

#### Fine-tuning

```shell
# Single GPU
python tools_opencd/train.py configs_DynamicVis/LEVIR-CD/name_to_config.py  # name_to_config.py is the config file you want to use
# Multi-GPU
sh tools_opencd/dist_train.sh configs_DynamicVis/LEVIR-CD/name_to_config.py ${GPU_NUM}  # name_to_config.py is the config file you want to use, GPU_NUM is the number of GPUs used
```

#### Testing

```shell
# Single GPU
python tools_opencd/test.py configs_DynamicVis/LEVIR-CD/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
# Multi-GPU
sh tools_opencd/dist_test.sh configs_DynamicVis/LEVIR-CD/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py is the config file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

</details>


### Image Retrieval

<details>

#### Config Files and Main Parameter Explanation

We provide the configuration files for image retrieval in the [image retrieval configuration file](retrieval/README.md) folder.


</details>



## Model Weights Download

You can download the pretrained weights from [Hugging Face](https://huggingface.co/KyanChen/DynamicVis).

- `b` and `l` represent the size of the model, corresponding to `base` and `large`, respectively.
- `wo-token-selection` indicates that the model does not use the selective region-aware architecture.
- `X-epoch` indicates the weights of the model at the `X`-th training epoch.



## FAQ

<details>


We list some common problems and their corresponding solutions here. If you find any problems missing, please feel free to submit a PR to enrich this list. If you cannot find help here, please use [issue](https://github.com/KyanChen/DynamicVis/issues) to seek help. Please fill in all the required information in the template, which will help us locate the problem more quickly.

### 1. Do I need to install the MM series packages?

We recommend that you do not install the MM series packages (such as MMDet), as we have included everything you need. If you install the MM series packages, you may encounter errors when running the code. If you encounter an error that the module has not been registered, please check:

- Whether the module is a package that needs to be installed, if so, install it
- Whether the MM series packages are installed, if so, uninstall them
- Whether `@MODELS.register_module()` is added before the class name, if not, add it
- Whether `from .xxx import xxx` is added in `__init__.py`, if not, add it
- Whether `custom_imports = dict(imports=['dynamicvis'], allow_failed_imports=False)` is added in the Config file, if not, add it

### 2. Solution to dist_train.sh: Bad substitution

If you encounter a `Bad substitution` error when running `dist_train.sh`, please use `bash dist_train.sh` to run the script.


</details>

## Acknowledgements

This project is built upon [OpenMMLab](https://openmmlab.com/codebase). We thank the OpenMMLab developers.

## Citation

If you use DynamicVis in your research, please cite:

```bibtex
@article{chen2025dynamicvis,
  title={DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding},
  author={Chen, Keyan and Liu, Chenyang and Chen, Bowen and Li, Wenyuan and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2503.16426},
  year={2025}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Contact

For further questions❓, feel free to contact us 👬