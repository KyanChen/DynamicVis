import argparse
import glob
import math
import os
import os.path as osp

import mmcv
import mmengine
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert massachusetts roads dataset to mmsegmentation format')
    parser.add_argument('--dataset_path', default='/mnt/search01/dataset/cky_data/MassachusettsRoads/tiff', help='massachusetts roads folder path')
    parser.add_argument('--tmp_dir', default='/mnt/search01/dataset/cky_data/MassachusettsRoads_512', help='path of the temporary directory')
    parser.add_argument('-nproc', '--num_proc', type=int, default=32, help='number of process')
    parser.add_argument('-o', '--out_dir', default='/mnt/search01/dataset/cky_data/MassachusettsRoads_512', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    # Original image of massachusetts roads dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    image = mmcv.imread(image_path)

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    if to_label:
        color_map = np.array([[0, 0, 0], [255, 255, 255]])
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        img_name = osp.basename(image_path).split('.')[0]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{img_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))

def multi_wrapper(args):
    return clip_big_image(*args)

def main():
    args = parse_args()

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', f'MassachusettsRoads_{args.clip_size}')
    else:
        out_dir = args.out_dir

    train_val_test = [
        'train',
        'train_labels',
        # 'val', 'val_labels',
        # 'test', 'test_labels'
    ]
    for file_split in train_val_test:
        if 'labels' in file_split:
            out_folder = osp.join(out_dir, 'ann_dir', f'{file_split.split("_")[0]}_{args.clip_size}')
        else:
            out_folder = osp.join(out_dir, 'img_dir', f'{file_split}_{args.clip_size}')
        print(f'Making directories for {out_folder}...')
        mkdir_or_exist(out_folder)
        img_path_list = glob.glob(os.path.join(dataset_path, f'{file_split}/*.tif*'))
        args_list = [(img_path, out_folder, args, True) if 'labels' in img_path else (img_path, out_folder, args, False) for img_path in img_path_list]
        if args.num_proc > 1:
            mmengine.track_parallel_progress(multi_wrapper, args_list, args.num_proc)
        else:
            mmengine.track_progress(multi_wrapper, args_list)

    print('Done!')


if __name__ == '__main__':
    main()
