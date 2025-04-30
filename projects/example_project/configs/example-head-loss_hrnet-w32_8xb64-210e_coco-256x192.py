# Directly inherit the entire recipe you want to use.
_base_ = 'mmpose::body_2d_keypoint/topdown_heatmap/coco/' \
         'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'

# This line is to import your own modules.
custom_imports = dict(imports='models')

# Modify the model to use your own head and loss.
_base_['model']['head'] = dict(
    type='ExampleHead',
    in_channels=32,
    out_channels=17,
    deconv_out_channels=None,
    loss=dict(type='ExampleLoss', use_target_weight=True),
    decoder=_base_['codec'])
