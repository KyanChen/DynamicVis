custom_imports = dict(imports='dynamicvis', allow_failed_imports=False)
default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10, by_epoch=True,
        max_keep_ckpts=5, save_last=True,
        save_best='single-label/f1-score',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
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


work_dir = 'work_dirs/UCMerced/dynamicvis_b_uc_mamba'
data_root = '/mnt/search01/dataset/cky_data/UC/UCMerced_LandUse'
code_root = '/mnt/search01/usr/chenkeyan/codes/dynamicvis'

pretrained_ckpt = 'work_dirs/fMoW/pretrain_dynamicvis_b_bf16_mamba_wo_token_reduction/epoch_200.pth'

batch_size = 32
base_lr = 0.0004
num_classes = 21
img_size = 512
dataset_type = 'RSClsDataset'
val_interval = 10


vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='dynamicvis', group='UCMerced', name=work_dir.split('/')[-1]))
                ]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)


optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
)

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
        begin=warmup_epochs
    )
]
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=val_interval)

data_preprocessor = dict(
    num_classes=num_classes,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmpretrain.DynamicVisBackbone',
        arch='b',
        frozen_stages=1,
        path_type='forward_reverse_mean',
        sampling_scale=dict(type='decay', val=0.1),
        global_token_cfg=dict(pos='none', num=0),
        is_softmax_on_x=False,
        img_size=img_size,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        spatial_token_keep_ratios=[1, 1, 1, 1],
        out_indices=(3,),
        out_type='avg_featmap',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_ckpt,
            prefix='backbone.'),
    ),
    head=dict(
        type='DynamicVisClsHead',
        num_classes=num_classes,
        in_channels=768,
        # load_balancing_loss_cfg=dict(type='decay', val=0.1),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.05, mode='original'),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=img_size,
        crop_ratio_range=(0.3, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='AutoAugment',
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_size, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

test_pipeline = val_pipeline

num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_name='UC',
        data_root=data_root,
        ann_file=code_root+'/datainfo/ucmerced/train.txt',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_name='UC',
        data_root=data_root,
        ann_file=code_root+'/datainfo/ucmerced/val.txt',
        pipeline=val_pipeline,
    )
)
test_dataloader = val_dataloader
val_evaluator = [
    dict(type='SingleLabelMetric', num_classes=num_classes),
    dict(type='Accuracy'),
    ]

test_evaluator = val_evaluator
val_cfg = dict()
test_cfg = dict()

tta_model = dict(type='AverageClsScoreTTA')
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='vertical'),
                dict(type='RandomFlip', prob=[1., 1.], direction=['horizontal', 'vertical']),
            ],
            [dict(type='PackInputs'),]
        ]
    )
]