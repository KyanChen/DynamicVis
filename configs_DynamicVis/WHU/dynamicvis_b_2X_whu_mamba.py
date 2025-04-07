custom_imports = dict(imports='dynamicvis', allow_failed_imports=False)
default_scope = 'mmseg'

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)

work_dir = 'work_dirs/WHU/dynamicvis_b_2X_whu_mamba'
data_root = '/mnt/nj-larc/dataset/cky_data/WHU'
code_root = '/mnt/nj-larc/usr/chenkeyan/codes/dynamicvis'
pretrained_ckpt = 'work_dirs/fMoW/pretrain_dynamicvis_b_bf16_mamba/epoch_200.pth'

batch_size = 8
base_lr = 0.001
img_size = 1024
crop_size = (img_size, img_size)
dataset_type = 'WHUSegDataset'
val_interval = 10

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10, by_epoch=True,
        max_keep_ckpts=5, save_last=True,
        save_best=['seg/iou_building'],
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=1, test_out_dir=work_dir + '/vis'),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='dynamicvis', group='WHU', name=work_dir.split('/')[-1]))
                ]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=1)

warmup_epochs = 5
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=warmup_epochs,
        eta_min_ratio=0.001,
    )
]

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    ),
)
train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=val_interval)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.DynamicVisBackbone',
        arch='b',
        # frozen_stages=1,
        path_type='forward_reverse_mean',
        sampling_scale=dict(type='decay', val=0.1),
        global_token_cfg=dict(pos='head', num=-1),
        is_softmax_on_x=True,
        img_size=img_size,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        spatial_token_keep_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3,),
        out_type='featmap',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_ckpt,
            prefix='backbone.'),
    ),
    neck=dict(
        type='mmdet.FPN',
        # in_channels=[128, 256, 512, 1024],
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_ckpt,
            prefix='pre_neck.'),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(img_size, img_size), stride=(img_size // 2, img_size // 2))
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512*2, 512*2),
        ratio_range=(0.3, 1.7),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=180),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=crop_size, pad_val=dict(img=tuple(bgr_mean), mask=0)),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512*2, 512*2), keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=tuple(bgr_mean), mask=0)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/image',
            seg_map_path='train/label'
        ),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/image',
            seg_map_path='test/label'
        ),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')


test_evaluator = val_evaluator
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(512 * 2, 512 * 2), keep_ratio=True, interpolation='bicubic'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='vertical'),
                dict(type='RandomFlip', prob=[1., 1.], direction=['horizontal', 'vertical']),
            ],
            [dict(type='LoadAnnotationsToBinary')],
            [dict(type='PackSegInputs')]
        ])
]
tta_model = dict(type='mmseg.SegTTAModel')
