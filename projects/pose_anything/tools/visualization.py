import os
import random

import matplotlib.pyplot as plt
import numpy as np

COLORS = ([255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255,
                                                     0], [170, 255,
                                                          0], [85, 255,
                                                               0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255,
                                        255], [0, 170,
                                               255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0,
                                                       170], [255, 0,
                                                              85], [255, 0, 0])


def plot_results(support_img,
                 query_img,
                 support_kp,
                 support_w,
                 query_kp,
                 query_w,
                 skeleton,
                 initial_proposals,
                 prediction,
                 radius=6,
                 out_dir='./heatmaps'):
    img_names = [
        img.split('_')[0] for img in os.listdir(out_dir)
        if str_is_int(img.split('_')[0])
    ]
    if len(img_names) > 0:
        name_idx = max([int(img_name) for img_name in img_names]) + 1
    else:
        name_idx = 0

    h, w, c = support_img.shape
    prediction = prediction[-1].cpu().numpy() * h
    support_img = (support_img - np.min(support_img)) / (
        np.max(support_img) - np.min(support_img))
    query_img = (query_img - np.min(query_img)) / (
        np.max(query_img) - np.min(query_img))

    for index, (img, w, keypoint) in enumerate(
            zip([support_img, query_img], [support_w, query_w],
                [support_kp, prediction])):
        f, axes = plt.subplots()
        plt.imshow(img)
        for k in range(keypoint.shape[0]):
            if w[k] > 0:
                kp = keypoint[k, :2]
                c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
                patch = plt.Circle(kp, radius, color=c)
                axes.add_patch(patch)
                axes.text(kp[0], kp[1], k)
                plt.draw()
        for limb_index, limb in enumerate(skeleton):
            kp = keypoint[:, :2]
            if limb_index > len(COLORS) - 1:
                c = [x / 255 for x in random.sample(range(0, 255), 3)]
            else:
                c = [x / 255 for x in COLORS[limb_index]]
            if w[limb[0]] > 0 and w[limb[1]] > 0:
                patch = plt.Line2D([kp[limb[0], 0], kp[limb[1], 0]],
                                   [kp[limb[0], 1], kp[limb[1], 1]],
                                   linewidth=6,
                                   color=c,
                                   alpha=0.6)
                axes.add_artist(patch)
        plt.axis('off')  # command for hiding the axis.
        name = 'support' if index == 0 else 'query'
        plt.savefig(
            f'./{out_dir}/{str(name_idx)}_{str(name)}.png',
            bbox_inches='tight',
            pad_inches=0)
        if index == 1:
            plt.show()
        plt.clf()
        plt.close('all')


def str_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
