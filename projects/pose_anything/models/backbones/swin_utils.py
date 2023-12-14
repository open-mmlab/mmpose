# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import numpy as np
import torch
from scipy import interpolate


def load_pretrained(config, model, logger):
    checkpoint = torch.load(config, map_location='cpu')
    ckpt_model = checkpoint['model']
    if any([True if 'encoder.' in k else False for k in ckpt_model.keys()]):
        ckpt_model = {
            k.replace('encoder.', ''): v
            for k, v in ckpt_model.items() if k.startswith('encoder.')
        }
        print('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        print('Detect non-pre-trained model, pass without doing anything.')

    checkpoint = remap_pretrained_keys_swin(model, ckpt_model, logger)
    msg = model.load_state_dict(ckpt_model, strict=False)
    print(msg)

    del checkpoint
    torch.cuda.empty_cache()


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with
    # fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if 'relative_position_bias_table' in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f'Error in loading {key}, passing......')
            else:
                if L1 != L2:
                    print(f'{key}: Interpolate relative_position_bias_table '
                          f'using geo.')
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q**(i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print('Original positions = %s' % str(x))
                    print('Target positions = %s' % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(
                            src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f_cubic(dx,
                                                 dy)).contiguous().view(-1, 1).
                            to(relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in checkpoint_model.keys() if 'relative_position_index' in k
    ]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [
        k for k in checkpoint_model.keys() if 'relative_coords_table' in k
    ]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # re-map keys due to name change
    rpe_mlp_keys = [k for k in checkpoint_model.keys() if 'rpe_mlp' in k]
    for k in rpe_mlp_keys:
        checkpoint_model[k.replace('rpe_mlp',
                                   'cpb_mlp')] = checkpoint_model.pop(k)

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if 'attn_mask' in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model
