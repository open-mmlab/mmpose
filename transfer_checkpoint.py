import copy

import torch
from mmcv import Config

from mmpose.models import build_posenet

cfg_file = 'configs/body/2d_kpt_sview_rgb_img/' \
           'topdown_heatmap/coco/hrformer_small_coco_256x192.py'
src_file = 'models/hrt_small_coco_256x192.pth'
out_file = src_file.replace('hrt', 'hrformer')

cfg = Config.fromfile(cfg_file)
model = cfg.model
model = build_posenet(model)
model_dict = model.state_dict()

src_dict = torch.load(src_file)
src_dict = src_dict['state_dict']

src_left = copy.copy(src_dict)
model_left = copy.copy(model_dict)

out_dict = {}
for key in model_dict.keys():
    if key in src_dict.keys():
        out_dict[key] = src_dict[key]
        src_left.pop(key)
        model_left.pop(key)

    elif 'attn.w_msa.' in key:
        model_key = key
        src_key = key.replace('attn.w_msa.', 'attn.attn.')
        if src_key in src_dict.keys():
            out_dict[model_key] = src_dict[src_key]
            src_left.pop(src_key)
            model_left.pop(model_key)
        elif '.proj.' in model_key:
            src_key = src_key.replace('.proj.', '.out_proj.')
            if src_key in src_dict.keys():
                out_dict[model_key] = src_dict[src_key]
                src_left.pop(src_key)
                model_left.pop(model_key)
        elif '.qkv.' in model_key:
            src_q = src_key.replace('.qkv.', '.q_proj.')
            src_k = src_key.replace('.qkv.', '.k_proj.')
            src_v = src_key.replace('.qkv.', '.v_proj.')
            tmp_q = src_dict[src_q]
            tmp_k = src_dict[src_k]
            tmp_v = src_dict[src_v]
            tmp_qkv = torch.cat([tmp_q, tmp_k, tmp_v], dim=0)
            out_dict[model_key] = tmp_qkv
            model_left.pop(model_key)
            src_left.pop(src_q)
            src_left.pop(src_k)
            src_left.pop(src_v)

# try to load
tmp = model.load_state_dict(out_dict)
print(tmp)

torch.save(out_dict, out_file)
t = 0
