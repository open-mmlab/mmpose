_base_ = ['./higher_hrnet48_coco_512x512.py']

# fp16 settings
fp16 = dict(loss_scale='dynamic')
