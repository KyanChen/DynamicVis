
import os
import h5py
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, io
import torch.nn.functional as F
from functools import partial
from .dataset_registry import register_dataset
from .utils import Unsqueeze, SelectChannels, AddMeanChannels


LABEL_NAMES = ['Oil palm plantation', 'Timber plantation', 'Other large-scale plantations', 'Grassland shrubland', 'Small-scale agriculture', 'Small-scale mixed plantation', 'Small-scale oil palm plantation', 'Mining', 'Fish pond', 'Logging', 'Secondary forest', 'Other']
LABEL_DICT = {'Oil palm plantation': 0, 'Timber plantation': 1, 'Other large-scale plantations': 2, 'Grassland shrubland': 3, 'Small-scale agriculture': 4, 'Small-scale mixed plantation': 5, 'Small-scale oil palm plantation': 6, 'Mining': 7, 'Fish pond': 8, 'Logging': 9, 'Secondary forest': 10, 'Other': 11}
LABEL_NAMES_MERGED = ['Plantation', 'Grassland shrubland', 'Smallholder agriculture', 'Other']
MERGED_LABEL_DICT = {'Plantation': 0, 'Grassland shrubland': 1, 'Smallholder agriculture': 2, 'Other': 3}

MERGE_LABEL_DICT = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3}


class ForestNet(Dataset):
    """
    Dataset class for ForestNet, introduced in:
    Irvin, J., Sheng, H., Ramachandran, N., Johnson-Yu, S., Zhou, S., Story, K., ... & Ng, A. Y. (2020).
    Forestnet: Classifying drivers of deforestation in indonesia using deep learning on satellite imagery.
    arXiv preprint arXiv:2011.05479. https://arxiv.org/pdf/2011.05479.pdf
    """
    def __init__(self, dataset_root, split, transform=None, merge_labels=False):
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform

        assert os.path.isdir(dataset_root), ('ForestNet data not found. '
                                             'Download dataset with `sh datasets/forestnet_download.sh`')

        # Load file names and labels
        split_df = pd.read_csv(os.path.join(dataset_root, f'{split}.csv'))
        self.filenames = split_df['example_path'].values

        if not merge_labels:
            self.labels = split_df['label'].values
            self.labels = [LABEL_DICT[l] for l in self.labels]
            self.label_names = LABEL_NAMES
        else:
            self.labels = split_df['merged_label'].values
            self.labels = [MERGED_LABEL_DICT[l] for l in self.labels]
            self.label_names = LABEL_NAMES_MERGED

        self.num_labels = len(self.label_names)
        self.labels = F.one_hot(torch.tensor(self.labels), self.num_labels)

    def __len__(self):
        return len(self.filenames)
        # return 4

    def __getitem__(self, index):
        # Load RGB and infrared data from png and npy file using the composite data
        visible = io.read_image(os.path.join(self.dataset_root, self.filenames[index], 'images', 'visible', 'composite.png'))
        infrared = torch.tensor(np.load(os.path.join(
            self.dataset_root, self.filenames[index], 'images', 'infrared', 'composite.npy')).astype(int))
        # Stack data into the expected order
        data = torch.concat([visible, infrared.permute(2, 0, 1)])
        # # Scale data from 0 - 255 (RGB) to 0 - 10000 (HLS)
        # data = data / 255 * 10000
        data = data.float()

        if self.transform is not None:
            data = self.transform(data)

        sample = {
            'image': data,
            'label': self.labels[index]
        }

        return sample


def init_forestnet(bands, normalize, merge_labels, cfg, *args, **kwargs):
    """
    Init m-ForestNet dataset.
    """
    # Get dataset parameters
    split = cfg['dataset']['split']

    # Get BigEarthNet directory
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    forestnet_dir = os.path.join(DATA_DIR, 'ForestNetDataset')

    # Init transforms
    image_transforms = [
        SelectChannels(bands),
        transforms.Resize(size=cfg['model']['img_size'], antialias=True),
    ]

    if normalize:
        if len(bands) != len(cfg['model']['data_mean']):
            # Add mean channels values for missing channels (e.g. for RGB data)
            image_transforms.append(AddMeanChannels(cfg['model']['data_mean']))
        # Normalize images
        if cfg['model']['data_mean'][0] < 1.1:
            mean = [m * 255. for m in cfg['model']['data_mean']]
            std = [s * 255. for s in cfg['model']['data_std']]
        else:
            mean = cfg['model']['data_mean']
            std = cfg['model']['data_std']
        image_transforms.append(transforms.Normalize(mean=mean, std=std))
        image_transforms.append(Unsqueeze(dim=1))  # add time dim

    # Init dataset
    dataset = ForestNet(
        dataset_root=forestnet_dir,
        split=split,
        transform=transforms.Compose(image_transforms),
        merge_labels=merge_labels,
    )

    return dataset


# Add datasets to the registry
register_dataset('ForestNet', partial(init_forestnet, [0, 1, 2, 3, 4, 5], True, False))

register_dataset('ForestNet4', partial(init_forestnet, [0, 1, 2, 3, 4, 5], True, True))

register_dataset('ForestNetRGB', partial(init_forestnet, [0, 1, 2], True, False))

register_dataset('ForestNet4RGB', partial(init_forestnet, [0, 1, 2], True, True))

register_dataset('ForestNetVisual', partial(init_forestnet, [2, 1, 0], False, False))

register_dataset('ForestNet4Visual', partial(init_forestnet, [2, 1, 0], False, True))
