custom_imports = dict(imports='dynamicvis', allow_failed_imports=False)
default_scope = 'mmdet'

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

work_dir = 'work_dirs/LevirShip/dynamicvis_b_levirship_mamba'
data_root = '/mnt/search01/dataset/cky_data/levir-ship'
code_root = '/mnt/search01/usr/chenkeyan/codes/dynamicvis'
pretrained_ckpt = 'work_dirs/fMoW/pretrain_dynamicvis_b_bf16_mamba/epoch_200.pth'

batch_size = 8
base_lr = 0.0002
find_unused_parameters = True
num_classes = 1
img_size = 1024
crop_size = (img_size, img_size)
dataset_type = 'LevirShipDetDataset'
val_interval = 5
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5, by_epoch=True,
        max_keep_ckpts=5, save_last=True,
        save_best=['coco/bbox_mAP'],
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False, score_thr=0.3, interval=1, test_out_dir=work_dir+'/vis')
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='dynamicvis', group='LevirShip', name=work_dir.split('/')[-1]))
                ]
line_width = 2
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', line_width=line_width)

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

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    ),
)
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=val_interval)

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='FCOS',
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
        type='FPN',
        # in_channels=[128, 256, 512, 1024],
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        # start_level=1,
        # add_extra_convs='on_output',  # use P5
        num_outs=5,
        # relu_before_extra_convs=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_ckpt,
            prefix='pre_neck.'),
    ),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=num_classes,
        # regress_ranges=((0, 40), (32, 80), (64, 160), (128, 1024)),
        regress_ranges=((0, 20), (16, 40), (32, 80), (64, 160), (128, 1024)),
        # regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1024)),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        # strides=[8, 16, 32, 64],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 1.8),
        resize_type='Resize',
        keep_ratio=True,
        interpolation='bicubic'
    ),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=tuple(bgr_mean))),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(32, 32)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=tuple(bgr_mean))),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor')
    )
]
num_workers = 8
persistent_workers = True
indices = None
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        ann_file=data_root+'/annotations/instances_train2017.json',
        data_prefix=dict(img='images/train2017'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        indices=indices,
        data_root=data_root,
        ann_file=data_root+'/annotations/instances_test2017.json',
        data_prefix=dict(img='images/test2017'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox'],
    format_only=False,
    iou_thrs=[0.5],
    ann_file=data_root + '/annotations/instances_test2017.json',
    backend_args=backend_args,
)

test_evaluator = val_evaluator
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=crop_size, keep_ratio=True)],
            [
                # ``RandomFlip`` must be placed before ``RandomCenterCropPad``,
                # otherwise bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=0.),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='vertical'),
                dict(type='RandomFlip', prob=[1., 1.], direction=['horizontal', 'vertical']),
            ],
            # [
            #     dict(
            #         type='RandomCenterCropPad',
            #         ratios=None,
            #         border=None,
            #         mean=[0, 0, 0],
            #         std=[1, 1, 1],
            #         to_rgb=True,
            #         test_mode=True,
            #         test_pad_mode=['logical_or', 31],
            #         test_pad_add_pix=1),
            # ],
            [dict(type='Pad', size=crop_size, pad_val=dict(img=tuple(bgr_mean), mask=0))],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'border'))
            ]
        ])
]

