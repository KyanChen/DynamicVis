custom_imports = dict(imports='dynamicvis', allow_failed_imports=False)
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1, by_epoch=True,
        max_keep_ckpts=5, save_last=True,
        save_best='single-label/f1-score',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmpretrain.VisualizationHook', enable=False),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)

work_dir = 'work_dirs/fMoW/pretrain_dynamicvis_b_bf16_mamba'

data_root = f'/mnt/nlp-ali/dataset/cky_data/fmow-rgb-tar'
code_root = f'/mnt/nlp-ali/usr/chenkeyan/codes/dynamicvis'

batch_size = 148
num_workers = 8
persistent_workers = True
non_blocking = True
prefetch_factor = 2
pin_memory = True

num_classes = 63
img_size = 512
val_interval = 10

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='dynamicvis', group='fMoW', name=work_dir.split('/')[-1]))
                ]

visualizer = dict(type='mmpretrain.UniversalVisualizer', vis_backends=vis_backends)

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=val_interval)

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    non_blocking=non_blocking,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# model settings
model = dict(
    type='mmpretrain.DynamicVisPretrainClassifier',
    backbone=dict(
        type='mmpretrain.DynamicVisBackbone',
        arch='b',
        path_type='forward_reverse_mean',
        sampling_scale=dict(type='fixed', val=0.1),
        global_token_cfg=dict(pos='head', num=-1),
        is_softmax_on_x=True,
        img_size=img_size,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        spatial_token_keep_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        out_type='featmap',
    ),
    pre_neck=dict(
        type='FPN',
        # in_channels=[128, 256, 512, 1024],
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    neck=dict(
        type='GenericRoIExtractor',
        aggregation='sum',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2, use_torchvision=True),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
        pre_cfg=dict(
            type='ConvModule',
            in_channels=256,
            out_channels=256,
            kernel_size=5,
            padding=2,
            inplace=False,
        ),
        post_cfg=dict(
            type='GeneralizedAttention',
            in_channels=256,
            spatial_range=-1,
            num_heads=6,
            attention_type='0100',
            kv_stride=2)
    ),
    head=dict(
        type='mmpretrain.DynamicVisPretrainClsHead',
        num_classes=num_classes,
        with_mil=True,
        in_channels=256,
        loss=dict(type='mmpretrain.LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
)

train_pipeline = [
    dict(type='mmpretrain.LoadImageFromImgbytes', to_float32=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=(img_size, img_size),
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(img_size, img_size),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=False),
    dict(type='Pad', size=(img_size, img_size), pad_val=dict(img=tuple(bgr_mean))),
    # dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='mmpretrain.LoadImageFromImgbytes', to_float32=True),
    dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
    dict(type='Pad', size=(img_size, img_size), pad_val=dict(img=tuple(bgr_mean))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(8, 8), keep_empty=True),
    dict(type='PackDetInputs')
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    sampler=None,
    dataset=dict(
        type='mmdet.PretrainFmowWebDataset',
        shards_path_or_url=data_root+'/pretrain_list/{00000..00127}.tar',
        data_name='Fmow',
        per_gpu_batch_size=1,  # we set per_gpu_batch_size=1 as we don't use a webdataloder
        num_workers=num_workers,
        pipeline=train_pipeline,
        shuffle_buffer_size=5000,
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    sampler=None,
    dataset=dict(
        type='mmdet.PretrainFmowWebDataset',
        shards_path_or_url=data_root+'/test_2w_list/{00000..00127}.tar',
        data_name='Fmow',
        per_gpu_batch_size=1,  # we set per_gpu_batch_size=1 as we don't use a webdataloder
        num_workers=num_workers,
        pipeline=test_pipeline,
        test_mode=True,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmpretrain.SingleLabelMetric',
    num_classes=num_classes,
)
test_evaluator = val_evaluator
val_cfg = dict()
test_cfg = dict()

base_lr = 0.0004

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        by_epoch=True,
        begin=0
    )
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    dtype='bfloat16',  # bfloat16 or float16
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    # clip_grad=dict(max_norm=20, norm_type=2),
)
runner_type = 'Runner'
