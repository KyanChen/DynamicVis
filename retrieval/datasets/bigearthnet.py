
import os
import torch
from torchvision import transforms
# from torchgeo.datasets import BigEarthNet
from .base_bigearthnet import BigEarthNet
from functools import partial
from .dataset_registry import register_dataset
from .utils import SelectChannels, Unsqueeze, DictTransforms, ConvertType, AddMeanChannels


def init_bigearthnet(bands, normalize, num_classes, cfg, *args, **kwargs):
    """
    Init BigEarthNet dataset, with S2 data and 43 classes as default.
    """
    # Get dataset parameters
    split = cfg['dataset']['split']
    satellite = cfg['dataset']['satellite'] if 'satellite' in cfg['dataset'] else 's2'

    # Get BigEarthNet directory
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    bigearthnet_dir = os.path.join(DATA_DIR, 'BigEarthNet')
    # Check if data is downloaded
    assert os.path.isdir(os.path.join(bigearthnet_dir, BigEarthNet.metadata['s2']['directory'])), \
        "Download BigEarthNet with `sh datasets/bigearthnet_download.sh` or specify the DATA_DIR via a env variable."

    # Init transforms
    image_transforms = [
        SelectChannels(bands),
        ConvertType(torch.float),
        transforms.Resize(size=cfg['model']['img_size'], antialias=True),
    ]

    if normalize:
        if len(bands) != len(cfg['model']['data_mean']):
            # Add mean channels values for missing channels (e.g. for BGR data)
            image_transforms.append(AddMeanChannels(cfg['model']['data_mean']))
        # Normalize images
        if cfg['model']['data_mean'][0] < 1.1:
            mean = [m * 5000 for m in cfg['model']['data_mean']]
            std = [s * 5000 for s in cfg['model']['data_std']]
        else:
            mean = cfg['model']['data_mean']
            std = cfg['model']['data_std']
        image_transforms.append(transforms.Normalize(mean=mean, std=std))
        image_transforms.append(Unsqueeze(dim=1))  # add time dim

    ben_transforms = DictTransforms({'image': transforms.Compose(image_transforms)})

    # Init dataset
    dataset = BigEarthNet(
        root=bigearthnet_dir,
        split=split,
        bands=satellite,
        num_classes=num_classes,
        transforms=ben_transforms,
        cache_dir=os.path.join(bigearthnet_dir, 'cached_data'),
    )

    return dataset


# Add dataset to the registry
# Using the six channels from Prithvi
register_dataset('BigEarthNet', partial(init_bigearthnet, [1, 2, 3, 8, 10, 11], True, 43))

register_dataset('BigEarthNetBGR', partial(init_bigearthnet, [1, 2, 3], True, 43))
register_dataset('BigEarthNetRGB', partial(init_bigearthnet, [3, 2, 1], True, 43))
register_dataset('BigEarthNetVisual', partial(init_bigearthnet, [3, 2, 1], False, 43))

register_dataset('BigEarthNet19', partial(init_bigearthnet, [1, 2, 3, 8, 10, 11], True, 19))

register_dataset('BigEarthNet19BGR', partial(init_bigearthnet, [1, 2, 3], True, 19))
register_dataset('BigEarthNet19RGB', partial(init_bigearthnet, [3, 2, 1], True, 19))
register_dataset('BigEarthNet19Visual', partial(init_bigearthnet, [3, 2, 1], False, 19))