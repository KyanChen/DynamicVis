import sys
import cv2
sys.path.append(sys.path[0] + '/../../..')
import mmcv
import numpy as np
from mmdet.datasets.transforms import Resize
import webdataset as wds
import glob
import json
import random
import os
import mmengine
import tqdm
from mmengine import fileio


def get_train_val_pairs(data_root, save_dir):
	for split in ['train', 'val', 'test', 'seq']:
		file_pairs = []
		if split in ['train', 'val']:
			json_files = mmengine.list_dir_or_file(
				data_root+'/'+split,
				list_dir=False,
				list_file=True,
				recursive=True,
				suffix='.json'
			)
			json_files = [data_root+'/'+split+'/'+x for x in json_files]
			img_files = [x.replace('.json', '.jpg') for x in json_files]
			file_pairs = list(zip(json_files, img_files))

		elif split in ['test', 'seq']:
			gt_mapping_file = f'{data_root}/{split}_gt_mapping.json'
			gt_mappings = fileio.load(gt_mapping_file)
			for gt_mapping in tqdm.tqdm(gt_mappings):
				input_folder = gt_mapping['input']  # seq_gt/fire_station/fire_station_168
				output_folder = gt_mapping['output']  # seq/0000019

				json_files = glob.glob(f'{data_root}/{input_folder}/*.json')
				img_file_folder = f'{data_root}/{output_folder}'
				img_file_names = [
					os.path.basename(x).replace('.json', '.jpg').replace(os.path.basename(os.path.dirname(x)), os.path.basename(output_folder)) \
					for x in json_files]
				img_files = [f'{img_file_folder}/{x}' for x in img_file_names]

				file_pairs += list(zip(json_files, img_files))

		with open(f'{save_dir}/{split}.txt', 'w') as f:
			for json_file, img_file in file_pairs:
				f.write(f'{json_file.replace(data_root+"/", "")} {img_file.replace(data_root+"/", "")}\n')
		print(f'{split} list saved to {save_dir}/{split}.txt')
		print(f'{len(file_pairs)} samples')

	# cat the train, val, seq, test files to get the full list
	test_files = mmengine.list_from_file(f'{save_dir}/test.txt')
	random.shuffle(test_files)
	# select 2W samples for test
	test_set_2w = test_files[:20000]
	with open(f'{save_dir}/test_2w_list.txt', 'w') as f:
		for line in test_set_2w:
			f.write(line+'\n')
	print(f'test 2w list saved to {save_dir}/test_2w_list.txt')
	print(f'{len(test_set_2w)} samples')

	# cat the train, val, seq, test files to get the full list
	pretrain_files = test_files[20000:]
	for split in ['train', 'val', 'seq']:
		pretrain_files += mmengine.list_from_file(f'{save_dir}/{split}.txt')
	random.shuffle(pretrain_files)
	with open(f'{save_dir}/pretrain_list.txt', 'w') as f:
		for line in pretrain_files:
			f.write(line+'\n')
	print(f'pretrain list saved to {save_dir}/pretrain_list.txt')
	print(f'{len(pretrain_files)} samples')

	'''
	train list saved to datainfo/fmow/train.txt
	727144 samples
	val list saved to datainfo/pretrain/val.txt
	106081 samples
	test list saved to datainfo/pretrain/test.txt
	106946 samples
	seq list saved to datainfo/pretrain/seq.txt
	107520 samples
	test 2w list saved to datainfo/pretrain/test_2w.txt
	20000 samples
	pretrain list saved to datainfo/pretrain/pretrain.txt
	1027691 samples
	'''

max_edge = 1024
resize_transform = Resize(scale=(max_edge, max_edge), keep_ratio=True)
def convert_img_gt(img_bytes, gt_data, img_name):
	img = mmcv.imfrombytes(img_bytes, flag='color', backend='cv2')
	gt_info = json.loads(gt_data)

	bounding_boxes = gt_info['bounding_boxes']
	gt_bboxes = []
	gt_bboxes_labels = []
	for i, ann in enumerate(bounding_boxes):
		if "raw_location" in ann:
			continue
		category = ann['category']
		x, y, w, h = ann['box']
		gt_bboxes.append([x, y, x + w, y + h])
		gt_bboxes_labels.append(category)

	data_info = dict(img=img)
	data_info['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
	data_info['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=str)

	if img.shape[0] > max_edge or img.shape[1] > max_edge:
		data_info = resize_transform(data_info)

	img_bytes = cv2.imencode('.png', data_info['img'])[1].tobytes()
	gt_data = dict(
		gt_bboxes=data_info['gt_bboxes'].tolist(),
		gt_bboxes_labels=data_info['gt_bboxes_labels'].tolist()
	)
	gt_data = json.dumps(gt_data)
	return img_bytes, gt_data


def save_as_tar(item):
	data_root, gt_img_files, out_file_path, idx_worker = item
	if idx_worker == 0:
		pbar = mmengine.ProgressBar(len(gt_img_files))
		pbar.start()

	sink = wds.TarWriter(out_file_path, encoder=False)

	for gt_img_file in gt_img_files:
		gt_name, img_name = gt_img_file.split(' ')

		img_path = data_root + '/' + img_name
		gt_path = data_root + '/' + gt_name

		if not os.path.exists(img_path) or not os.path.exists(gt_path):
			print(f'File not found: {img_path} or {gt_path}')
			continue
		img_bytes = fileio.get(img_path, backend_args=None)
		gt_data = json.dumps(mmengine.load(gt_path))

		img_bytes, gt_data = convert_img_gt(img_bytes, gt_data, os.path.basename(img_path))


		sample = {
			'__key__': img_name,
			'jpg': img_bytes,
			'json': gt_data.encode('utf-8')
		}
		sink.write(sample)
		if idx_worker == 0:
			pbar.update()
	sink.close()



if __name__ == '__main__':
	data_root = f'/mnt/search01/dataset/cky_data/fmow-rgb'
	meta_data_save_dir = 'datainfo/fmow'
	tar_save_dir = 'data/fmow-tar'
	n_process = 1  # n_process = n_shards
	n_shards = n_process

	mmengine.mkdir_or_exist(meta_data_save_dir)
	mmengine.mkdir_or_exist(tar_save_dir)

	# get train, val, test, seq pairs list
	get_train_val_pairs(data_root, meta_data_save_dir)

	key = input("Press q to exit, others to continue: ")
	if key == 'q':
		exit()
	# save as tar
	for train_test_file in ['pretrain_list.txt', 'test_2w_list.txt']:
		out_dir_tmp = f'{tar_save_dir}/{os.path.splitext(train_test_file)[0]}'
		mmengine.mkdir_or_exist(out_dir_tmp)
		items = mmengine.list_from_file(f'{meta_data_save_dir}/{train_test_file}')
		num_samples = len(items)
		print(f'num_samples: {num_samples}')
		meta_info = dict(
			num_samples=num_samples,
			num_shards=n_shards,
		)
		mmengine.dump(meta_info, f'{out_dir_tmp}/meta.json')

		random.shuffle(items)
		# split items to n_shards
		items = np.array_split(items, n_shards)
		items = [(data_root, list(x), f'{out_dir_tmp}/{idx:05d}.tar', idx) for idx, x in enumerate(items)]

		if n_process > 1:
			results = mmengine.track_parallel_progress(save_as_tar, items, n_process)
		else:
			results = mmengine.track_progress(save_as_tar, items)




