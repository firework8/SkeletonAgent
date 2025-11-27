modality = 'b'
graph = 'uav'
work_dir = f'./work_dirs/uav_csv2/b'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='GCN_Module_LLM',
        llm_model='gpt4o', llm_modality = modality, num_classes=155,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(
        type='SimpleHead', joint_cfg=graph, num_classes=155, in_channels=384, work_dir=work_dir, interval_epoch=5, weight_1=0.3, weight_2=0.2))

dataset_type = 'PoseDataset'
ann_file = '/data/uav/uav_human.pkl'
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='uav', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='uav', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='uav', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])    
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='csv2_train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='csv2_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='csv2_val'))

# setting: 4 GPU  64  0.1  ->  1 GPU  64/4=16  0.1/4=0.025
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 150
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
