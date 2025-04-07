import json
import sys
import webdataset as wds
import tqdm

data_root = '/mnt/nlp-ali/dataset/cky_data/fmow-rgb'

train_shards_path_or_url = data_root + '-tar' + '/pretrain_list/{00000..00127}.tar'
test_shards_path_or_url = data_root + '-tar' + '/test_2w_list/{00000..00127}.tar'

num_box = 0
num_img = 0
dataset = wds.WebDataset(train_shards_path_or_url)

t_bar = tqdm.tqdm(total=1027691)
for sample in dataset:
	box_data = json.loads(sample['jpg.json'])
	gt_bboxes = box_data['gt_bboxes']
	num_box += len(gt_bboxes)
	num_img += 1
	t_bar.update(1)
t_bar.close()
print(f'num_box: {num_box}')
print(f'num_img: {num_img}')