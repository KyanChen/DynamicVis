custom_imports = dict(imports='dynamicvis', allow_failed_imports=False)
default_scope = 'opencd'

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

work_dir = 'work_dirs/LEVIR-CD/dynamicvis_l_2X_levircd_mamba'
data_root = '/mnt/search01/dataset/cky_data/levir-cd'
code_root = '/mnt/search01/usr/chenkeyan/codes/dynamicvis'
pretrained_ckpt = 'work_dirs/fMoW/pretrain_dynamicvis_l_bf16_mamba/best_single-label_f1-score_epoch_150.pth'

batch_size = 2
base_lr = 0.0002
input_size = 1024
dataset_type = 'LEVIR_CD_Dataset'
val_interval = 10

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10, by_epoch=True,
        max_keep_ckpts=5, save_last=True,
        save_best=['cd/iou_changed'],
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', draw=False, interval=1,
                       # test_out_dir=work_dir + '/vis'
                       ),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='dynamicvis', group='LEVIR-CD', name=work_dir.split('/')[-1]))
                ]
visualizer = dict(type='CDLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=1)

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

bgr_mean = [123.675, 116.28, 103.53]
bgr_std = [58.395, 57.12, 57.375]
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=bgr_mean * 2,
    std=bgr_std * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32)
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.DynamicVisBackbone',
        arch='l',
        # frozen_stages=1,
        path_type='forward_reverse_mean',
        sampling_scale=dict(type='decay', val=0.1),
        global_token_cfg=dict(pos='head', num=-1),
        is_softmax_on_x=True,
        img_size=input_size,
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
        type='SequentialNeck',
        necks=[
            dict(
                type='DualInputsFPN',
                # in_channels=[96, 192, 384, 768],
                in_channels=[128, 256, 512, 1024],
                out_channels=256,
                num_outs=5,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=pretrained_ckpt,
                    prefix='pre_neck.'),
            ),
            dict(
                type='DualInputsSimpleFusionNeck',
                in_channels=[256, 256, 256, 256, 256],
                return_tuple=True,
            ),
            dict(
                type='mmseg.UNetDecodeNeck',
                in_channels=[256] * 5,
                dec_num_convs=(2, 2, 2, 2),
                dec_dilations=(1, 1, 1, 1),
                with_cp=False,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                upsample_cfg=dict(type='mmseg.InterpConv'),
            )
        ]
    ),
    decode_head=dict(
        type='MLPSegHead',
        out_size=(input_size // 4, input_size // 4),
        in_channels=[256] * 5,
        in_index=[0, 1, 2, 3, 4],
        channels=256,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='mmseg.DiceLoss', loss_weight=3.0)
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(input_size, input_size), stride=(input_size // 2, input_size // 2))
)


train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(
        type='MultiImgRandomResize',
        scale=(1024*2, 1024*2),
        ratio_range=(0.2, 1.8),
        keep_ratio=True,
        interpolation='bicubic'
    ),
    dict(type='MultiImgRandomCrop', crop_size=(input_size, input_size), cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPad', size=(input_size, input_size)),
    dict(type='MultiImgPackSegInputs')
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(2*1024, 2*1024), keep_ratio=True, interpolation='bicubic'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
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
            seg_map_path='train/label',
            img_path_from='train/A',
            img_path_to='train/B'),
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
            seg_map_path='test/label',
            img_path_from='test/A',
            img_path_to='test/B'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='CDMetric')


test_evaluator = val_evaluator
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
	dict(type='MultiImgLoadImageFromFile', backend_args=None),
    dict(type='MultiImgResize', scale=(2 * 1024, 2 * 1024), keep_ratio=True, interpolation='bicubic'),
    dict(
		type='TestTimeAug',
		transforms=[
			[
				dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
				for r in img_ratios
			],
			[
				dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
				dict(type='MultiImgRandomFlip', prob=1., direction='horizontal'),
				dict(type='MultiImgRandomFlip', prob=1., direction='vertical'),
                # dict(type='MultiImgRandomFlip', prob=[1., 1.], direction=['horizontal', 'vertical']),
			],
			[dict(type='MultiImgLoadAnnotations')],
			[dict(type='MultiImgPackSegInputs')]
		])
]
tta_model = dict(type='mmseg.SegTTAModel')
