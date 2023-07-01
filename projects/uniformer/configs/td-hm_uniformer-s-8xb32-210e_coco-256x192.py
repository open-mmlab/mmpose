_base_ = ['./_base_/td-hm_uniformer-b-8xb32-210e_coco-256x192']

model = dict(
    # pretrained='/path/to/hrt_small.pth', # Set the path to pretrained backbone here
    backbone=dict(
        layers=[3, 4, 8, 3],
        drop_path_rate=0.2
    ))