_base_ = ['./td-hm_res50_8xb64-210e_coco-256x192.py']

# fp16 settings
fp16 = dict(loss_scale='dynamic')
