# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmpose.registry import HOOKS
from mmpose.structures import PoseDataSample, merge_data_samples


@HOOKS.register_module()
class PoseVisualizationHook(Hook):
    """Pose Estimation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``out_dir`` is specified, it means that the prediction results
        need to be saved to ``out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        enable (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 enable: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 out_dir: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk')):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PoseDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)

        self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = data_batch[0]['data_sample'].get('img_path')
        img_bytes = self.file_client.get(img_path)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        data_sample = outputs[0]
        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=data_sample,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                show=self.show,
                wait_time=self.wait_time,
                kpt_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PoseDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.get('img_path')
            img_bytes = self.file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_sample = merge_data_samples([data_sample])

            out_file = None
            if self.out_dir is not None:
                out_file_name, postfix = os.path.basename(img_path).split('.')
                index = len([
                    fname for fname in os.listdir(self.out_dir)
                    if fname.startswith(out_file_name)
                ])
                out_file = f'{out_file_name}_{index}.{postfix}'
                out_file = os.path.join(self.out_dir, out_file)

            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                wait_time=self.wait_time,
                kpt_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)
