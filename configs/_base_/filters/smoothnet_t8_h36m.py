filter_cfg = dict(
    type='SmoothNetFilter',
    window_size=8,
    output_size=8,
    checkpoint='https://download.openmmlab.com/mmpose/plugin/smoothnet/'
    'smoothnet_ws8_h36m.pth',
    hidden_size=512,
    res_hidden_size=256,
    num_blocks=3)
