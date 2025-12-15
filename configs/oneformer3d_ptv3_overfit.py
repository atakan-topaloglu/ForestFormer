"""
ForestFormer PTV3 - Overfitting Test Config

This config is designed to overfit to a single training sample to verify
that the model can learn. If the model cannot overfit to one sample,
there's a bug in the architecture or training pipeline.

Expected behavior:
- Loss should decrease rapidly
- After ~100-200 iterations, loss should be near zero
- Predictions should match ground truth perfectly on the training sample

Usage:
    python tools/train.py configs/oneformer3d_ptv3_overfit.py
"""

_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# Model settings - same as main config
num_channels = 64
num_instance_classes = 3
num_semantic_classes = 3
radius = 16
grid_size = 0.2

model = dict(
    type='ForAINetV2OneFormer3D_PTV3',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3,
    num_channels=num_channels,
    grid_size=grid_size,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    stuff_classes=[0],
    thing_cls=[1, 2],
    radius=radius,
    backbone=dict(
        type='PTV3Backbone',
        in_channels=3,
        grid_size=grid_size,
        out_channels=num_channels,
        pretrained=None,  # No pretrained weights for overfit test
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 256, 256, 256, 256),  # Reduced for memory
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(256, 256, 256, 256),  # Reduced for memory
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,  # No drop path for overfitting
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ForAINet",),
    ),
    decoder=dict(
        type='ForAINetv2QueryDecoder',
        num_layers=6,
        num_classes=1,
        num_instance_queries=300,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,  # No dropout for overfitting
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='ForAINetv2UnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=0.2),
        inst_criterion=dict(
            type='InstanceCriterionForAI',
            matcher=dict(
                type='HungarianMatcher',
                costs=[
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)]),
            loss_weight=[1.0, 1.0, 0.5],
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=250,
        inst_score_thr=0.4,
        pan_score_thr=0.4,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0],
        thing_cls=[0]))

# Dataset settings
dataset_type = 'ForAINetV2SegDataset_'
data_root_forainetv2 = 'data/ForAINetV2/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

# Training pipeline - NO augmentation for overfitting
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius, deterministic=True),
    dict(type='GridSample', grid_size=grid_size, deterministic=True),
    dict(
        type='PointSample_',
        num_points=640000,
        deterministic=True),  # Reduced for faster iteration
    dict(type='SkipEmptyScene_'),
    dict(type='PointInstClassMapping_',
        num_classes=num_instance_classes),
    # NO augmentation - we want to overfit
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask', 'ratio_inspoint'
        ])
]

# Validation pipeline - same as train but for inference
val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius, deterministic=True),
    dict(type='GridSample', grid_size=grid_size, deterministic=True),
    dict(
        type='PointSample_',
        num_points=640000,
        deterministic=True),
    dict(type='PointInstClassMapping_',
        num_classes=num_instance_classes),
    dict(type='Pack3DDetInputs_', keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask'])
]

# Dataloader - single sample, repeated
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # No shuffle - same sample
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        filter_empty_gt=True,
        box_type_3d='Depth',
        backend_args=None,
        indices=[0]))  # Only use first sample!

# Validation on the SAME sample to track metrics
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_train.pkl',  # Same as train!
        data_prefix=data_prefix,
        pipeline=val_pipeline,
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None,
        indices=[0]))  # Same single sample!

test_dataloader = val_dataloader

# Evaluation settings
class_names = ['ground', 'wood', 'leaf']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],
    classes=class_names,
    dataset_name='ForAINetV2')

sem_mapping = [0, 1, 2]
inst_mapping = sem_mapping[1:]
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=list(range(1, num_semantic_classes)),
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

# Optimizer - higher learning rate for faster convergence
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0),  # Higher LR, no weight decay
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        T_max=1000,     # total epochs
        eta_min=1e-5   # final LR
    )
]

# Hooks - log frequently to monitor loss
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    # Fix spconv weight format for SpConvUNet during validation + checkpoint saving
    # NOTE: PTV3Backbone is automatically skipped (doesn't have this issue)
    # For SpConvUNet: fixes validation metrics AND saves checkpoints correctly
    dict(type='SpConvWeightFixHook', verbose=True),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=500,  # Save every 500 iterations
        max_keep_ckpts=2,
        save_optimizer=True),
    logger=dict(type='LoggerHook', interval=1),  # Log EVERY iteration
    visualization=dict(type='Det3DVisualizationHook', draw=False))

# Visualization
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Training settings - iteration based, short run
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1000,  # 1000 iterations should be enough to overfit
    val_interval=100)  # Validate every 100 iterations

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
find_unused_parameters = True

# Print model info
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
