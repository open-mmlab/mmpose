_base_ = '../../rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'

model = dict(test_cfg=dict(nms_pre=1, score_thr=0.0, max_per_img=1))
