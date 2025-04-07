
import logging
import os
import torch
import subprocess
from .model_registry import MODEL_REGISTRY


def build_model(cfg):
    # Get model class from registry
    model_name = cfg['model']['name']
    logging.info(f'Load model {model_name}')
    assert model_name in MODEL_REGISTRY, (f'model {model_name} not registered.'
                                          f'Select a model from {MODEL_REGISTRY.keys()}')
    model_constructor = MODEL_REGISTRY[model_name]

    # Init model
    model = model_constructor(**cfg['model'])
    logging.debug('Model initialized')

    return model
