# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.visualization import Visualizer


class OpencvBackendVisualizer(Visualizer):
    """Base visualizer with opencv backend support.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.
        backend (str): Backend used to draw elements on the image and display
            the image. Defaults to 'matplotlib'.
        alpha (int, float): The transparency of bboxes. Defaults to ``1.0``
    """

    def __init__(self,
                 name='visualizer',
                 backend: str = 'matplotlib',
                 *args,
                 **kwargs):
        super().__init__(name, *args, **kwargs)
        assert backend in ('opencv', 'matplotlib'), f'the argument ' \
            f'\'backend\' must be either \'opencv\' or \'matplotlib\', ' \
            f'but got \'{backend}\'.'
        self.backend = backend

    @master_only
    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
            backend (str): The backend to save the image.
        """
        assert image is not None
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        if self.backend == 'matplotlib':
            # add a small 1e-2 to avoid precision lost due to matplotlib's
            # truncation (https://github.com/matplotlib/matplotlib/issues/15363)  # noqa
            self.fig_save.set_size_inches(  # type: ignore
                (self.width + 1e-2) / self.dpi,
                (self.height + 1e-2) / self.dpi)
            # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
            self.ax_save.cla()
            self.ax_save.axis(False)
            self.ax_save.imshow(
                image,
                extent=(0, self.width, self.height, 0),
                interpolation='none')

    @master_only
    def get_image(self) -> np.ndarray:
        """Get the drawn image. The format is RGB.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        assert self._image is not None, 'Please set image using `set_image`'
        if self.backend == 'matplotlib':
            return super().get_image()
        else:
            return self._image

    @master_only
    def draw_circles(self,
                     center: Union[np.ndarray, torch.Tensor],
                     radius: Union[np.ndarray, torch.Tensor],
                     face_colors: Union[str, tuple, List[str],
                                        List[tuple]] = 'none',
                     alpha: float = 1.0,
                     **kwargs) -> 'Visualizer':
        """Draw single or multiple circles.

        Args:
            center (Union[np.ndarray, torch.Tensor]): The x coordinate of
                each line' start and end points.
            radius (Union[np.ndarray, torch.Tensor]): The y coordinate of
                each line' start and end points.
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of circles. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value,
                all the lines will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of circles.
                Defaults to 0.8.
        """
        if self.backend == 'matplotlib':
            super().draw_circles(
                center=center,
                radius=radius,
                face_colors=face_colors,
                alpha=alpha,
                **kwargs)
        elif self.backend == 'opencv':
            if isinstance(face_colors, str):
                face_colors = mmcv.color_val(face_colors)

            if alpha == 1.0:
                self._image = cv2.circle(self._image,
                                         (int(center[0]), int(center[1])),
                                         int(radius), face_colors, -1)
            else:
                img = cv2.circle(self._image.copy(),
                                 (int(center[0]), int(center[1])), int(radius),
                                 face_colors, -1)
                self._image = cv2.addWeighted(self._image, 1 - alpha, img,
                                              alpha, 0)
        else:
            raise ValueError(f'got unsupported backend {self.backend}')

    @master_only
    def draw_texts(
        self,
        texts: Union[str, List[str]],
        positions: Union[np.ndarray, torch.Tensor],
        font_sizes: Optional[Union[int, List[int]]] = None,
        colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        vertical_alignments: Union[str, List[str]] = 'top',
        horizontal_alignments: Union[str, List[str]] = 'left',
        bboxes: Optional[Union[dict, List[dict]]] = None,
        **kwargs,
    ) -> 'Visualizer':
        """Draw single or multiple text boxes.

        Args:
            texts (Union[str, List[str]]): Texts to draw.
            positions (Union[np.ndarray, torch.Tensor]): The position to draw
                the texts, which should have the same length with texts and
                each dim contain x and y.
            font_sizes (Union[int, List[int]], optional): The font size of
                texts. ``font_sizes`` can have the same length with texts or
                just single value. If ``font_sizes`` is single value, all the
                texts will have the same font size. Defaults to None.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors
                of texts. ``colors`` can have the same length with texts or
                just single value. If ``colors`` is single value, all the
                texts will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            vertical_alignments (Union[str, List[str]]): The verticalalignment
                of texts. verticalalignment controls whether the y positional
                argument for the text indicates the bottom, center or top side
                of the text bounding box.
                ``vertical_alignments`` can have the same length with
                texts or just single value. If ``vertical_alignments`` is
                single value, all the texts will have the same
                verticalalignment. verticalalignment can be 'center' or
                'top', 'bottom' or 'baseline'. Defaults to 'top'.
            horizontal_alignments (Union[str, List[str]]): The
                horizontalalignment of texts. Horizontalalignment controls
                whether the x positional argument for the text indicates the
                left, center or right side of the text bounding box.
                ``horizontal_alignments`` can have
                the same length with texts or just single value.
                If ``horizontal_alignments`` is single value, all the texts
                will have the same horizontalalignment. Horizontalalignment
                can be 'center','right' or 'left'. Defaults to 'left'.
            font_families (Union[str, List[str]]): The font family of
                texts. ``font_families`` can have the same length with texts or
                just single value. If ``font_families`` is single value, all
                the texts will have the same font family.
                font_familiy can be 'serif', 'sans-serif', 'cursive', 'fantasy'
                or 'monospace'.  Defaults to 'sans-serif'.
            bboxes (Union[dict, List[dict]], optional): The bounding box of the
                texts. If bboxes is None, there are no bounding box around
                texts. ``bboxes`` can have the same length with texts or
                just single value. If ``bboxes`` is single value, all
                the texts will have the same bbox. Reference to
                https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
                for more details. Defaults to None.
            font_properties (Union[FontProperties, List[FontProperties]], optional):
                The font properties of texts. FontProperties is
                a ``font_manager.FontProperties()`` object.
                If you want to draw Chinese texts, you need to prepare
                a font file that can show Chinese characters properly.
                For example: `simhei.ttf`, `simsun.ttc`, `simkai.ttf` and so on.
                Then set ``font_properties=matplotlib.font_manager.FontProperties(fname='path/to/font_file')``
                ``font_properties`` can have the same length with texts or
                just single value. If ``font_properties`` is single value,
                all the texts will have the same font properties.
                Defaults to None.
                `New in version 0.6.0.`
        """  # noqa: E501

        if self.backend == 'matplotlib':
            super().draw_texts(
                texts=texts,
                positions=positions,
                font_sizes=font_sizes,
                colors=colors,
                vertical_alignments=vertical_alignments,
                horizontal_alignments=horizontal_alignments,
                bboxes=bboxes,
                **kwargs)

        elif self.backend == 'opencv':
            font_scale = max(0.1, font_sizes / 30)
            thickness = max(1, font_sizes // 15)

            text_size, text_baseline = cv2.getTextSize(texts,
                                                       cv2.FONT_HERSHEY_DUPLEX,
                                                       font_scale, thickness)

            x = int(positions[0])
            if horizontal_alignments == 'right':
                x = max(0, x - text_size[0])
            y = int(positions[1])
            if vertical_alignments == 'top':
                y = min(self.height, y + text_size[1])

            if bboxes is not None:
                bbox_color = bboxes[0]['facecolor']
                if isinstance(bbox_color, str):
                    bbox_color = mmcv.color_val(bbox_color)

                y = y - text_baseline // 2
                self._image = cv2.rectangle(
                    self._image, (x, y - text_size[1] - text_baseline // 2),
                    (x + text_size[0], y + text_baseline // 2), bbox_color,
                    cv2.FILLED)

            self._image = cv2.putText(self._image, texts, (x, y),
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                      colors, thickness - 1)
        else:
            raise ValueError(f'got unsupported backend {self.backend}')

    @master_only
    def draw_bboxes(self,
                    bboxes: Union[np.ndarray, torch.Tensor],
                    edge_colors: Union[str, tuple, List[str],
                                       List[tuple]] = 'g',
                    line_widths: Union[Union[int, float],
                                       List[Union[int, float]]] = 2,
                    **kwargs) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw with
                the format of(x1,y1,x2,y2).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of bboxes.
                Defaults to 0.8.
        """
        if self.backend == 'matplotlib':
            super().draw_bboxes(
                bboxes=bboxes,
                edge_colors=edge_colors,
                line_widths=line_widths,
                **kwargs)

        elif self.backend == 'opencv':
            self._image = mmcv.imshow_bboxes(
                self._image,
                bboxes,
                edge_colors,
                top_k=-1,
                thickness=line_widths,
                show=False)
        else:
            raise ValueError(f'got unsupported backend {self.backend}')

    @master_only
    def draw_lines(self,
                   x_datas: Union[np.ndarray, torch.Tensor],
                   y_datas: Union[np.ndarray, torch.Tensor],
                   colors: Union[str, tuple, List[str], List[tuple]] = 'g',
                   line_widths: Union[Union[int, float],
                                      List[Union[int, float]]] = 2,
                   **kwargs) -> 'Visualizer':
        """Draw single or multiple line segments.

        Args:
            x_datas (Union[np.ndarray, torch.Tensor]): The x coordinate of
                each line' start and end points.
            y_datas (Union[np.ndarray, torch.Tensor]): The y coordinate of
                each line' start and end points.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors of
                lines. ``colors`` can have the same length with lines or just
                single value. If ``colors`` is single value, all the lines
                will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
        """
        if self.backend == 'matplotlib':
            super().draw_lines(
                x_datas=x_datas,
                y_datas=y_datas,
                colors=colors,
                line_widths=line_widths,
                **kwargs)

        elif self.backend == 'opencv':

            self._image = cv2.line(
                self._image, (x_datas[0], y_datas[0]),
                (x_datas[1], y_datas[1]),
                colors,
                thickness=line_widths)
        else:
            raise ValueError(f'got unsupported backend {self.backend}')

    @master_only
    def draw_polygons(self,
                      polygons: Union[Union[np.ndarray, torch.Tensor],
                                      List[Union[np.ndarray, torch.Tensor]]],
                      edge_colors: Union[str, tuple, List[str],
                                         List[tuple]] = 'g',
                      alpha: float = 1.0,
                      **kwargs) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            polygons (Union[Union[np.ndarray, torch.Tensor],\
                List[Union[np.ndarray, torch.Tensor]]]): The polygons to draw
                with the format of (x1,y1,x2,y2,...,xn,yn).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of polygons. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value,
                all the lines will have the same colors. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
                Defaults to 'g.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of polygons.
                Defaults to 0.8.
        """
        if self.backend == 'matplotlib':
            super().draw_polygons(
                polygons=polygons,
                edge_colors=edge_colors,
                alpha=alpha,
                **kwargs)

        elif self.backend == 'opencv':
            if alpha == 1.0:
                self._image = cv2.fillConvexPoly(self._image, polygons,
                                                 edge_colors)
            else:
                img = cv2.fillConvexPoly(self._image.copy(), polygons,
                                         edge_colors)
                self._image = cv2.addWeighted(self._image, 1 - alpha, img,
                                              alpha, 0)
        else:
            raise ValueError(f'got unsupported backend {self.backend}')

    @master_only
    def show(self,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: float = 0.,
             continue_key=' ') -> None:
        """Show the drawn image.

        Args:
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
        """
        if self.backend == 'matplotlib':
            super().show(
                drawn_img=drawn_img,
                win_name=win_name,
                wait_time=wait_time,
                continue_key=continue_key)

        elif self.backend == 'opencv':
            # Keep images are shown in the same window, and the title of window
            # will be updated with `win_name`.
            if not hasattr(self, win_name):
                self._cv_win_name = win_name
                cv2.namedWindow(winname=f'{id(self)}')
                cv2.setWindowTitle(f'{id(self)}', win_name)
            else:
                cv2.setWindowTitle(f'{id(self)}', win_name)
            shown_img = self.get_image() if drawn_img is None else drawn_img
            cv2.imshow(str(id(self)), mmcv.bgr2rgb(shown_img))
            cv2.waitKey(int(np.ceil(wait_time * 1000)))
        else:
            raise ValueError(f'got unsupported backend {self.backend}')
