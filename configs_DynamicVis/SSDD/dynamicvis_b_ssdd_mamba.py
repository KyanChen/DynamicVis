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

work_dir = 'work_dirs/SSDD/dynamicvis_b_ssdd_mamba'
data_root = '/mnt/search01/dataset/cky_data/SSDD'
code_root = '/mnt/search01/usr/chenkeyan/codes/dynamicvis'
pretrained_ckpt = 'work_dirs/fMoW/pretrain_dynamicvis_b_bf16_mamba/epoch_200.pth'

batch_size = 8
base_lr = 0.0001
find_unused_parameters = True
num_classes = 1
img_size = 1024
crop_size = (img_size, img_size)
dataset_type = 'SSDDInsSegDataset'
val_interval = 10

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10, by_epoch=True,
        max_keep_ckpts=5, save_last=True,
        save_best=['coco/bbox_mAP', 'coco/segm_mAP'],
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False, score_thr=0.3, interval=1, test_out_dir=work_dir+'/vis')
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='dynamicvis', group='SSDD', name=work_dir.split('/')[-1]))
                ]
line_width = 2
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', line_width=line_width, alpha=0.8)

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
train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=val_interval)

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='MaskRCNN',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.DynamicVisBackbone',
        arch='b',
        frozen_stages=1,
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
        num_outs=5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_ckpt,
            prefix='pre_neck.'),
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
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
                kv_stride=2),
            init_cfg=dict(
                type='Pretrained',
                checkpoint=pretrained_ckpt,
                prefix='neck.'),
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2, use_torchvision=True),
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
                kv_stride=2),
            init_cfg=dict(
                type='Pretrained',
                checkpoint=pretrained_ckpt,
                prefix='neck.'),
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))
)

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.1, 2.0),
        resize_type='Resize',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=crop_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=tuple(bgr_mean), mask=0)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=tuple(bgr_mean), mask=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
        ann_file=code_root + '/datainfo/ssdd/SSDD_instances_train.json',
        data_prefix=dict(img='imgs'),
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
        ann_file=code_root + '/datainfo/ssdd/SSDD_instances_val.json',
        data_prefix=dict(img='imgs'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args,
    ann_file=code_root + '/datainfo/ssdd/SSDD_instances_val.json',
)

test_evaluator = val_evaluator
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
