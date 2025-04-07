import argparse
import glob
import math
import os
import os.path as osp
import shutil

import mmcv
import numpy as np
from mmengine.utils import ProgressBar


# copy the label
split = 'test'

from_folder = '/Users/kyanchen/Downloads/Test Labels' if split == 'test' else '/Users/kyanchen/Downloads/OSCD/Train Labels'
output_folder = f'/Users/kyanchen/Downloads/OSCD/{split}/label'
os.makedirs(output_folder, exist_ok=True)

imgs = glob.glob(os.path.join(from_folder, '*/*/*-cm.tif'))
prog_bar = ProgressBar(len(imgs))
for img in imgs:
    new_path = os.path.join(
        output_folder,
        os.path.basename(os.path.dirname(os.path.dirname(img)))+'.png')
    img = mmcv.imread(img, flag='unchanged')
    nb_classes = np.unique(img)
    print(f'nb_classes: {nb_classes}')
    img[img > 1] = 255
    img[img <= 1] = 0
    mmcv.imwrite(img, new_path)
    prog_bar.update()



# imgs
referencing_folder = f'/Users/kyanchen/Downloads/OSCD/{split}/label'
referencing_names = glob.glob(os.path.join(referencing_folder, '*.png'))
referencing_names = [os.path.basename(name).split('.')[0] for name in referencing_names]

from_folder = '/Users/kyanchen/Downloads/Images'
output_folder = f'/Users/kyanchen/Downloads/OSCD/{split}'
os.makedirs(output_folder, exist_ok=True)

prog_bar = ProgressBar(len(referencing_names))
for referencing_name in referencing_names:
    imgA = from_folder + '/' + referencing_name + '/pair' + '/img1.png'
    imgB = from_folder + '/' + referencing_name + '/pair' + '/img2.png'
    new_pathA = os.path.join(
        output_folder,
        'A', referencing_name + '.png')
    new_pathB = os.path.join(
        output_folder,
        'B', referencing_name + '.png')
    os.makedirs(os.path.dirname(new_pathA), exist_ok=True)
    os.makedirs(os.path.dirname(new_pathB), exist_ok=True)
    shutil.copy(imgA, new_pathA)
    shutil.copy(imgB, new_pathB)
    prog_bar.update()