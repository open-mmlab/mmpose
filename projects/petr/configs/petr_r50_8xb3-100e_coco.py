_base_ = ['petr_r50_8xb4-100e_coco.py']

auto_scale_lr = dict(base_batch_size=24)
train_dataloader = dict(batch_size=3)
optim_wrapper = dict(optimizer=dict(lr=0.00015))
