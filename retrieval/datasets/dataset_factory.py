
import logging
from .dataset_registry import DATASET_REGISTRY
from torchvision import transforms


def load_dataset(cfg):
    # load settings from cfg
    dataset_name = cfg['dataset']['name']
    logging.info(f"Load dataset {dataset_name} ({cfg['dataset']['split']} split)")
    assert dataset_name in DATASET_REGISTRY, (f"Dataset {dataset_name} not registered. "
                                              f"Select a dataset from {DATASET_REGISTRY.keys()} ")
    # get dataset fc from registry
    dataset_fn = DATASET_REGISTRY[dataset_name]
    # load dataset
    dataset = dataset_fn(cfg)
    return dataset
