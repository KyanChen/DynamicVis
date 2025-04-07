import glob
import os.path
import tqdm
import cv2
import mmengine
import numpy as np

folder = '/Users/chenkeyan/Documents/visualizations/whu-building'
save_folder = '/Users/chenkeyan/Documents/visualizations/whu-building/vis_data'
mmengine.mkdir_or_exist(save_folder)

sub_folders = glob.glob(folder + '/*')
for sub_folder in tqdm.tqdm(sub_folders):
	img_list = glob.glob(sub_folder + '/*.tif')
	model_name = os.path.basename(sub_folder).split('_')[1]
	for img_file in img_list:
		img = cv2.imread(img_file)
		ori_img, gt, pred = img[:, :512], img[:, 512:1024], img[:, 1024:]
		vis_data = np.zeros_like(gt)
		gt = gt[:, :, 0]
		pred = pred[:, :, 0]
		# true positive in green
		vis_data[(gt > 128) & (pred > 128)] = [0, 255, 0]
		# false positive in red
		vis_data[(gt < 128) & (pred > 128)] = [0, 0, 255]
		# false negative in blue
		vis_data[(gt > 128) & (pred < 128)] = [255, 0, 0]
		# true negative in black
		vis_data[(gt < 128) & (pred < 128)] = [0, 0, 0]

		# calculate the iou score
		tp = np.sum((gt > 128) & (pred > 128))
		fp = np.sum((gt < 128) & (pred > 128))
		fn = np.sum((gt > 128) & (pred < 128))
		tn = np.sum((gt < 128) & (pred < 128))
		iou = tp / (tp + fp + fn)
		img_name = os.path.basename(img_file).split('.')[0]
		cv2.imwrite(f'{save_folder}/{img_name}.png', ori_img)
		cv2.imwrite(f'{save_folder}/{img_name}_gt.png', gt)
		cv2.imwrite(f'{save_folder}/{img_name}_{model_name}_{np.round(iou, 2)}.png', vis_data)