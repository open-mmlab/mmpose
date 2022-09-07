_base_ = ['./td-hm_res50_8xb64-210e_coco-256x192.py']

# fp16 settings
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
)
