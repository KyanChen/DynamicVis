import json
import os
import sys

import mmcv
import numpy as np

import webdataset as wds
import tqdm

data_root = '/mnt/nlp-ali/dataset/cky_data/fmow-rgb'

train_shards_path_or_url = data_root + '-tar' + '/pretrain_list/{00000..00127}.tar'
test_shards_path_or_url = data_root + '-tar' + '/test_2w_list/{00000..00127}.tar'
vis_data_folder = 'work_dirs/vis_img_ann'
if not os.path.exists(vis_data_folder):
	os.makedirs(vis_data_folder)


dataset = wds.WebDataset(train_shards_path_or_url).shuffle(10000)

idx = 0
max_iter = 10000
for sample in dataset:
	if idx % 20 == 0:
		img_bytes = sample['jpg.jpg']
		gt_data = json.loads(sample['jpg.json'])
		gt_bboxes = gt_data['gt_bboxes']
		gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
		gt_bboxes_labels = '_'.join(gt_data['gt_bboxes_labels'])
		img = mmcv.imfrombytes(img_bytes, flag='color')
		mmcv.imshow_bboxes(img, gt_bboxes, colors='red', show=False, thickness=2, out_file=f'{vis_data_folder}/{idx}_{gt_bboxes_labels}.png')
	idx += 1
	if idx > max_iter:
		break