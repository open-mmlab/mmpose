# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToPILImage


class SimCCVisualizer:

    def draw_instance_xy_heatmap(self,
                                 heatmap: torch.Tensor,
                                 overlaid_image: Optional[np.ndarray],
                                 n: int = 20,
                                 mix: bool = True,
                                 weight: float = 0.5):
        """Draw heatmaps of GT or prediction.

        Args:
            heatmap (torch.Tensor): Tensor of heatmap.
            overlaid_image (np.ndarray): The image to draw.
            n (int): Number of keypoint, up to 20.
            mix (bool):Whether to merge heatmap and original image.
            weight (float): Weight of original image during fusion.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        heatmap2d = heatmap.data.max(0, keepdim=True)[0]
        xy_heatmap, K = self.split_simcc_xy(heatmap)
        K = K if K <= n else n
        blank_size = tuple(heatmap.size()[1:])
        maps = {'x': [], 'y': []}
        for i in xy_heatmap:
            x, y = self.draw_1d_heatmaps(i['x']), self.draw_1d_heatmaps(i['y'])
            maps['x'].append(x)
            maps['y'].append(y)
        white = self.creat_blank(blank_size, K)
        map2d = self.draw_2d_heatmaps(heatmap2d)
        if mix:
            map2d = cv.addWeighted(overlaid_image, 1 - weight, map2d, weight,
                                   0)
        self.image_cover(white, map2d, int(blank_size[1] * 0.1),
                         int(blank_size[0] * 0.1))
        white = self.add_1d_heatmaps(maps, white, blank_size, K)
        return white

    def split_simcc_xy(self, heatmap: Union[np.ndarray, torch.Tensor]):
        """Extract one-dimensional heatmap from two-dimensional heatmap and
        calculate the number of keypoint."""
        size = heatmap.size()
        k = size[0] if size[0] <= 20 else 20
        maps = []
        for _ in range(k):
            xy_dict = {}
            single_heatmap = heatmap[_]
            xy_dict['x'], xy_dict['y'] = self.merge_maps(single_heatmap)
            maps.append(xy_dict)
        return maps, k

    def merge_maps(self, map_2d):
        """Synthesis of one-dimensional heatmap."""
        x = map_2d.data.max(0, keepdim=True)[0]
        y = map_2d.data.max(1, keepdim=True)[0]
        return x, y

    def draw_1d_heatmaps(self, heatmap_1d):
        """Draw one-dimensional heatmap."""
        size = heatmap_1d.size()
        length = max(size)
        np_heatmap = ToPILImage()(heatmap_1d).convert('RGB')
        cv_img = cv.cvtColor(np.asarray(np_heatmap), cv.COLOR_RGB2BGR)
        if size[0] < size[1]:
            cv_img = cv.resize(cv_img, (length, 15))
        else:
            cv_img = cv.resize(cv_img, (15, length))
        single_map = cv.applyColorMap(cv_img, cv.COLORMAP_JET)
        return single_map

    def creat_blank(self,
                    size: Union[list, tuple],
                    K: int = 20,
                    interval: int = 10):
        """Create the background."""
        blank_height = int(
            max(size[0] * 2, size[0] * 1.1 + (K + 1) * (15 + interval)))
        blank_width = int(
            max(size[1] * 2, size[1] * 1.1 + (K + 1) * (15 + interval)))
        blank = np.zeros((blank_height, blank_width, 3), np.uint8)
        blank.fill(255)
        return blank

    def draw_2d_heatmaps(self, heatmap_2d):
        """Draw a two-dimensional heatmap fused with the original image."""
        np_heatmap = ToPILImage()(heatmap_2d).convert('RGB')
        cv_img = cv.cvtColor(np.asarray(np_heatmap), cv.COLOR_RGB2BGR)
        map_2d = cv.applyColorMap(cv_img, cv.COLORMAP_JET)
        return map_2d

    def image_cover(self, background: np.ndarray, foreground: np.ndarray,
                    x: int, y: int):
        """Paste the foreground on the background."""
        fore_size = foreground.shape
        background[y:y + fore_size[0], x:x + fore_size[1]] = foreground
        return background

    def add_1d_heatmaps(self,
                        maps: dict,
                        background: np.ndarray,
                        map2d_size: Union[tuple, list],
                        K: int,
                        interval: int = 10):
        """Paste one-dimensional heatmaps onto the background in turn."""
        y_startpoint, x_startpoint = [int(1.1*map2d_size[1]),
                                      int(0.1*map2d_size[0])],\
                                     [int(0.1*map2d_size[1]),
                                      int(1.1*map2d_size[0])]
        x_startpoint[1] += interval * 2
        y_startpoint[0] += interval * 2
        add = interval + 10
        for i in range(K):
            self.image_cover(background, maps['x'][i], x_startpoint[0],
                             x_startpoint[1])
            cv.putText(background, str(i),
                       (x_startpoint[0] - 30, x_startpoint[1] + 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            self.image_cover(background, maps['y'][i], y_startpoint[0],
                             y_startpoint[1])
            cv.putText(background, str(i),
                       (y_startpoint[0], y_startpoint[1] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            x_startpoint[1] += add
            y_startpoint[0] += add
        return background[:x_startpoint[1] + y_startpoint[1] +
                          1, :y_startpoint[0] + x_startpoint[0] + 1]
