import itertools
import json
import math
import os
import warnings
from typing import List, Union
import mmengine
import numpy as np
from braceexpand import braceexpand
from mmengine.dataset import Compose
import tqdm

from mmdet.registry import DATASETS as DET_DATASETS
from mmseg.registry import DATASETS as SEG_DATASETS
from mmpretrain.datasets import BaseDataset as PretrainBaseDataset
from mmdet.datasets import BaseDetDataset, CocoDataset
from mmseg.datasets import BaseSegDataset
from .category_map import CATEGORIES
import webdataset as wds
from mmpretrain.registry import DATASETS as PRETRAIN_DATASETS
from opencd.datasets import _BaseCDDataset
from opencd.registry import DATASETS as CDDATASETS


@PRETRAIN_DATASETS.register_module()
class RSClsDataset(PretrainBaseDataset):
    def __init__(self,
                 data_root: str = '',
                 data_name: str = 'AID',
                 load_to_memory: bool = False,
                 ann_file: str = '',
                 **kwargs):
        metainfo = {'classes': CATEGORIES[data_name]}
        self.load_to_memory = load_to_memory
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        img_files = mmengine.list_from_file(self.ann_file)
        data_list = []
        if self.load_to_memory:
            load_pipeline = self.pipeline.transforms[0]
            self.pipeline.transforms = self.pipeline.transforms[1:]
        for img_file in tqdm.tqdm(img_files):
            img_info = dict(
                img_path=self.data_root + '/' + img_file,
                gt_label=self.class_to_idx[img_file.split('/')[0]],
                redis_key=img_file
            )
            if self.load_to_memory:
                results = load_pipeline(img_info)
                img_info.update(results)
            data_list.append(img_info)
        return data_list

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        results = self.pipeline(data_info)
        return results



@DET_DATASETS.register_module()
class PretrainFmowWebDataset(wds.DataPipeline):
    def __init__(
            self,
            shards_path_or_url: Union[str, List[str]],
            data_name: str = "Fmow",
            pipeline: List[dict] = None,
            per_gpu_batch_size: int = 1,
            num_workers: int = 0,
            shuffle_buffer_size: int = 1000,
            test_mode: bool = False,
    ):
        if not isinstance(shards_path_or_url, str):
            shards_path_or_url = [list(braceexpand(urls)) for urls in shards_path_or_url]
            # flatten list using itertools
            shards_path_or_url = list(itertools.chain.from_iterable(shards_path_or_url))
        self.shards_path_or_url = shards_path_or_url
        self.test_mode = test_mode

        self.metainfo = {'classes': CATEGORIES[data_name]}
        self.cat2label = {cat: i for i, cat in enumerate(self.metainfo['classes'])}
        self.transform_pipeline = Compose(pipeline)

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(shards_path_or_url),
            # wds.SimpleShardList(shards_path_or_url),  # if use ResampledShards, it will shuffle the shards, and split by node and worker
            # wds.shuffle(100),
            # wds.split_by_node,
            # wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle_buffer_size) if not test_mode else None,
            wds.map(self.transform),
        ]
        super().__init__(*pipeline)

        num_gpu = mmengine.dist.get_world_size()

        global_batch_size = per_gpu_batch_size * num_gpu
        num_batches = math.ceil(self.real_len() / global_batch_size)
        num_workers = max(1, num_workers)
        self.num_worker_batches = math.ceil(num_batches / num_workers)  # per dataloader worker
        self.num_batches = self.num_worker_batches * num_workers
        self.num_samples = self.num_batches * global_batch_size

        self.with_length(self.num_samples // num_gpu)  # In Dataloader, if a iterable dataset, the length is defined as the length of the iterable dataset divided by bs
        self.with_epoch(self.num_worker_batches)  # multiple of per_gpu_batch_size as we loop the batch in external dataloader

    def transform(self, sample):
        '''
        'jpg': img_bytes,
        'json': gt_data
        '''
        sample_key = sample["__key__"]
        img_bytes = sample['jpg.jpg']
        gt_data = json.loads(sample['jpg.json'])
        gt_bboxes = gt_data['gt_bboxes']
        gt_bboxes_labels = [self.cat2label[x] for x in gt_data['gt_bboxes_labels']]
        data_info = dict(
            img_path=sample_key,
            img_bytes=img_bytes,
            gt_bboxes=np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4)),
            gt_bboxes_labels=np.array(gt_bboxes_labels, dtype=np.int64)
        )
        results = self.transform_pipeline(data_info)
        return results

    def real_len(self):
        meta_file = os.path.dirname(self.shards_path_or_url)+'/meta.json'
        if not os.path.exists(meta_file):
            warnings.warn(f"meta file {meta_file} not found")
            num_samples = 10000
        else:
            num_samples = mmengine.load(meta_file)['num_samples']
        if mmengine.dist.is_main_process():
            print(f"real_len: {num_samples}")
        return num_samples




@DET_DATASETS.register_module()
class NWPUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['airplane', 'ship', 'storage_tank', 'baseball_diamond',
                    'tennis_court', 'basketball_court', 'ground_track_field',
                    'harbor', 'bridge', 'vehicle'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255)]
    }


@DET_DATASETS.register_module()
class WHUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['building'],
        'palette': [(0, 255, 0)]
    }


@DET_DATASETS.register_module()
class SSDDInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['ship'],
        'palette': [(0, 0, 255)]
    }


@DET_DATASETS.register_module()
class LevirShipDetDataset(CocoDataset):
    METAINFO = {
        'classes': ['ship'],
        'palette': [(255, 0, 0)]
    }



@SEG_DATASETS.register_module()
class WHUSegDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])
    def __init__(
            self,
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def __getitem__(self, item):
        results = super().__getitem__(item)
        return results


@SEG_DATASETS.register_module()
class MassachusettsRoadsSegDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [255, 255, 255]])
    def __init__(
            self,
            img_suffix='.tiff',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def __getitem__(self, item):
        results = super().__getitem__(item)
        return results


@CDDATASETS.register_module()
class OSCD_CD_Dataset(_BaseCDDataset):
    """OSCD dataset"""
    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='',
                 seg_map_suffix='',
                 format_seg_map=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)