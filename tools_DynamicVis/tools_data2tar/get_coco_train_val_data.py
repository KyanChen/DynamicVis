import io
import sys

import PIL
import cv2
sys.path.append(sys.path[0] + '/../..')
import mmcv
import numpy as np
from mmdet.datasets.transforms import Resize
import webdataset as wds
import glob
import json
import random
import os
import mmengine
from tqdm import tqdm
from mmengine import fileio


def save_as_tar(item):
	gt_img_files, out_file_path, idx_worker = item
	if idx_worker == 0:
		pbar = mmengine.ProgressBar(len(gt_img_files))
		pbar.start()

	sink = wds.TarWriter(out_file_path, encoder=False)
	num_samples = 0
	for gt_img_file in gt_img_files:
		'''
		data_item = {
			'file_name': img_info['file_name'],
			'img_id': img_info['id'],
			'split': split,
			'img_path': img_path,
			'width': img_info['width'],
			'height': img_info['height'],
			'instances': instances,
		}
		'''
		img_path = gt_img_file['img_path']
		img_name = gt_img_file['file_name']
		gt_data = json.dumps(gt_img_file)

		if not os.path.exists(img_path):
			print(f'File not found: {img_path}')
			continue
		img_bytes = fileio.get(img_path, backend_args=None)
		try:
			img = np.asarray(PIL.Image.open(io.BytesIO(img_bytes)))
		except Exception as e:
			print(f'Error: {img_path}, {e}')
			continue
		sample = {
			'__key__': img_name,
			'png': img_bytes,
			'json': gt_data.encode('utf-8')
		}
		sink.write(sample)
		num_samples += 1
		if idx_worker == 0:
			pbar.update()
	sink.close()
	# 返回样本数目
	return num_samples


def read_coco_rows(coco_ann_path, coco_dataset_path):

	with open(coco_ann_path, "r") as f:
		coco_ann = json.load(f)
	id2name = {x['id']: x['name'] for x in coco_ann['categories']}
	name2id = {x['name']: x['id'] for x in coco_ann['categories']}
	cats = [id2name[x] for x in sorted(id2name)]
	print(cats)
	# import ipdb; ipdb.set_trace()
	# 构建image_id到annotations的映射
	image_ann_map = {}
	for ann in coco_ann['annotations']:
		image_id = ann['image_id']
		if image_id not in image_ann_map:
			image_ann_map[image_id] = []
		image_ann_map[image_id].append(ann)

	# 收集有效的图像数据
	image_data = []
	for img_info in tqdm(coco_ann['images'], desc=f'Processing {split} images'):
		img_path = os.path.join(coco_dataset_path, 'inpainting', img_info['file_name'])

		if not os.path.exists(img_path):
			print(f"Warning: {img_path} not exists, skip")
			continue

		annotations = image_ann_map.get(img_info['id'], [])
		if len(annotations) == 0:
			print(f"Warning: {img_path} has no annotations, skip")
		instances = []
		for i, ann in enumerate(annotations):
			instance = {}

			if ann.get('ignore', False):
				continue
			x1, y1, w, h = ann['bbox']
			inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
			inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
			if inter_w * inter_h == 0:
				continue
			if ann['area'] <= 0 or w < 1 or h < 1:
				continue
			if ann['category_id'] not in id2name:
				continue
			bbox = [x1, y1, x1 + w, y1 + h]

			if ann.get('iscrowd', False):
				instance['ignore_flag'] = 1
			else:
				instance['ignore_flag'] = 0
			instance['bbox'] = bbox
			instance['bbox_label'] = ann['category_id']

			if ann.get('segmentation', None):
				instance['mask'] = ann['segmentation']

			instances.append(instance)

		data_item = {
			'file_name': img_info['file_name'],
			'img_id': img_info['id'],
			'split': split,
			'img_path': img_path,
			'width': img_info['width'],
			'height': img_info['height'],
			'instances': instances,
		}
		image_data.append(data_item)

	return image_data

if __name__ == '__main__':
	coco_dataset_path = "/mnt/ali-sh-1/usr/qiming/detection"
	tar_save_dir = "/mnt/ali-sh-1/usr/qiming/detection/data/tar"
	n_process = 128  # n_process = n_shards
	n_shards = n_process

	mmengine.mkdir_or_exist(tar_save_dir)

	key = input("Press q to exit, others to continue: ")
	if key == 'q':
		exit()
	# save as tar
	for split in ['train', 'test']:
		out_dir_tmp = f'{tar_save_dir}/{split}'
		mmengine.mkdir_or_exist(out_dir_tmp)
		coco_ann_path = os.path.join(coco_dataset_path, f"{split}_ann_.json")
		items = read_coco_rows(coco_ann_path, coco_dataset_path)

		random.shuffle(items)
		# split items to n_shards
		items = np.array_split(items, n_shards)
		# 建立索引文件
		index_file = {
			"wids_version": 1,
			"shardlist": [],
			"name": f"coco_road_{split}",
		}

		items = [(list(x), f'{out_dir_tmp}/{idx:05d}.tar', idx) for idx, x in enumerate(items)]

		if n_process > 1:
			results = mmengine.track_parallel_progress(save_as_tar, items, n_process)
		else:
			results = mmengine.track_progress(save_as_tar, items)

		for idx, x in enumerate(results):
			index_file['shardlist'].append({
				'url': f'{idx:05d}.tar',
				'nsamples': x,
			})
			print('idx: {}, nsamples: {}'.format(idx, x))
		mmengine.dump(index_file, f'{out_dir_tmp}/index.json')

		num_samples = sum([x for x in results])
		print(f'num_samples: {num_samples}')
		meta_info = dict(
			num_samples=num_samples,
			num_shards=n_shards,
		)
		mmengine.dump(meta_info, f'{out_dir_tmp}/meta.json')




