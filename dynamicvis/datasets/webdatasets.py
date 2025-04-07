import glob
import itertools
import json
import logging
import math
import os
import warnings
from typing import List, Union

import mmengine
import torch
from braceexpand import braceexpand
from mmengine import print_log, is_abs, join_path
from mmengine.dataset import Compose, BaseDataset

from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS
from mmdet.registry import DATASETS as MMDET_DATASETS
import webdataset as wds


@MMDET_DATASETS.register_module()
class TarDetDataset(wds.DataPipeline):
    METAINFO = {
        'classes': (
            'object',
        ),
        'palette': [
            (0, 0, 255)
        ]
    }
    def __init__(
            self,
            shards_path_or_url: Union[str, List[str]],
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

        self.metainfo = self.METAINFO
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
        # print(sample.keys())
        img_bytes = sample['png.png']
        gt_data = json.loads(sample['png.json'])
        gt_data['img_bytes'] = img_bytes
        results = self.transform_pipeline(gt_data)
        if results is None:
            print(f'{sample_key} is None')
            return None
        if results['data_samples'].gt_instances is not None:
            results['data_samples'].gt_instances.labels = results['data_samples'].gt_instances.labels - 1
        return results

    def real_len(self):
        # return 128
        meta_file = os.path.dirname(self.shards_path_or_url)+'/meta.json'
        if not os.path.exists(meta_file):
            warnings.warn(f"meta file {meta_file} not found")
            num_samples = 10000
        else:
            num_samples = mmengine.load(meta_file)['num_samples']
        if mmengine.dist.is_main_process():
            print(f"real_len: {num_samples}")
        return num_samples


@MMDET_DATASETS.register_module()
class TarTestDetDataset(BaseDataset):
    METAINFO = {
        'classes': (
            'object',
        ),
        'palette': [
            (0, 0, 255)
        ]
    }

    def __init__(self,
                 shards_path_or_url: Union[str, List[str]],
                 pipeline: List[dict] = None,
                 test_mode: bool = True,
                 *args,
                 **kwargs) -> None:
        self.shards_path_or_url = shards_path_or_url
        super().__init__(pipeline=pipeline, test_mode=test_mode, *args, **kwargs)


    def load_data_list(self):
        print('load test data to memory')
        dataset = list(wds.WebDataset(self.shards_path_or_url, nodesplitter=None))
        print(f'data len: {len(dataset)}')
        return dataset

    def prepare_data(self, idx):
        sample = self.get_data_info(idx)
        sample_key = sample["__key__"]

        img_bytes = sample['png.png']
        gt_data = json.loads(sample['png.json'])
        gt_data['img_bytes'] = img_bytes
        data = self.pipeline(gt_data)
        if data is None:
            print(f'{sample_key} is None')
            return None
        if data['data_samples'].gt_instances is not None:
            data['data_samples'].gt_instances.labels = data['data_samples'].gt_instances.labels - 1
        return data