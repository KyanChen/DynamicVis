import json
import os
import warnings
from typing import Optional
import mmengine
import mmengine.fileio as fileio
import numpy as np
import mmcv
from mmcv import LoadImageFromFile, LoadAnnotations
from mmpretrain.registry import TRANSFORMS as PRETRAIN_TRANSFORMS
from mmseg.registry import TRANSFORMS as SEG_TRANSFORMS
from mmseg.datasets import LoadAnnotations as SEG_LoadAnnotations


@PRETRAIN_TRANSFORMS.register_module()
class LoadImageFromImgbytes(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        img_bytes = results['img_bytes']
        try:
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@SEG_TRANSFORMS.register_module()
class LoadAnnotationsToBinary(SEG_LoadAnnotations):
    def __init__(self,
                 format_seg_map='to_binary',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.format_seg_map = format_seg_map

    def _load_seg_map(self, results: dict) -> None:
        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        if self.format_seg_map is not None:
            if self.format_seg_map == 'to_binary':
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
                gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1
            else:
                raise ValueError('Invalid value {}'.format(results['format_seg_map']))
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')


