filter_cfg = dict(
    type='SmoothNetFilter',
    window_size=16,
    output_size=16,
    checkpoint='https://download.openmmlab.com/mmpose/plugin/smoothnet/'
    'smoothnet_ws16_h36m.pth',
    hidden_size=512,
    res_hidden_size=256,
    num_blocks=3)
