_base_ = ['../top_down/resnet/coco/res50_coco_256x192.py']

# fp16 settings
fp16 = dict(loss_scale=512.)
