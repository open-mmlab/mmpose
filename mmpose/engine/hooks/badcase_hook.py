# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os
import warnings
from typing import Dict, Optional, Sequence

import mmcv
import mmengine
import mmengine.fileio as fileio
import torch
from mmengine.config import ConfigDict
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmpose.registry import HOOKS, METRICS, MODELS
from mmpose.structures import PoseDataSample, merge_data_samples


@HOOKS.register_module()
class BadCaseAnalysisHook(Hook):
    """Bad Case Analyze Hook. Used to visualize validation and testing process
    prediction results.

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
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        interval (int): The interval of visualization. Defaults to 50.
        kpt_thr (float): The threshold to visualize the keypoints.
            Defaults to 0.3.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        metric_type (str): the mretic type to decide a badcase,
            loss or accuracy.
        metric (ConfigDict): The config of metric.
        metric_key (str): key of needed metric value in the return dict
            from class 'metric'.
        badcase_thr (float): min loss or max accuracy for a badcase.
    """

    def __init__(
        self,
        enable: bool = False,
        show: bool = False,
        wait_time: float = 0.,
        interval: int = 50,
        kpt_thr: float = 0.3,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
        metric_type: str = 'loss',
        metric: ConfigDict = ConfigDict(type='KeypointMSELoss'),
        metric_key: str = 'PCK',
        badcase_thr: float = 5,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args

        self.metric_type = metric_type
        if metric_type not in ['loss', 'accuracy']:
            raise KeyError(
                f'The badcase metric type {metric_type} is not supported by '
                f"{self.__class__.__name__}. Should be one of 'loss', "
                f"'accuracy', but got {metric_type}.")
        self.metric = MODELS.build(metric) if metric_type == 'loss'\
            else METRICS.build(metric)
        self.metric_name = metric.type if metric_type == 'loss'\
            else metric_key
        self.metric_key = metric_key
        self.badcase_thr = badcase_thr
        self.results = []

    def check_badcase(self, data_batch, data_sample):
        """Check whether the sample is a badcase.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        Return:
            is_badcase (bool): whether the sample is a badcase or not
            metric_value (float)
        """
        if self.metric_type == 'loss':
            gts = data_sample.gt_instances.keypoints
            preds = data_sample.pred_instances.keypoints
            weights = data_sample.gt_instances.keypoints_visible
            with torch.no_grad():
                metric_value = self.metric(
                    torch.from_numpy(preds), torch.from_numpy(gts),
                    torch.from_numpy(weights)).item()
            is_badcase = metric_value >= self.badcase_thr
        else:
            self.metric.process([data_batch], [data_sample.to_dict()])
            metric_value = self.metric.evaluate(1)[self.metric_key]
            is_badcase = metric_value <= self.badcase_thr
        return is_badcase, metric_value

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PoseDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if not self.enable:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.get('img_path')
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_sample = merge_data_samples([data_sample])

            is_badcase, metric_value = self.check_badcase(
                data_batch, data_sample)

            if is_badcase:
                img_name, postfix = os.path.basename(img_path).rsplit('.', 1)
                bboxes = data_sample.gt_instances.bboxes.astype(int).tolist()
                bbox_info = 'bbox' + str(bboxes)
                metric_postfix = self.metric_name + str(round(metric_value, 2))

                self.results.append({
                    'img': img_name,
                    'bbox': bboxes,
                    self.metric_name: metric_value
                })

                badcase_name = f'{img_name}_{bbox_info}_{metric_postfix}'

                out_file = None
                if self.out_dir is not None:
                    out_file = f'{badcase_name}.{postfix}'
                    out_file = os.path.join(self.out_dir, out_file)

                # draw gt keypoints in blue color
                self._visualizer.kpt_color = 'blue'
                self._visualizer.link_color = 'blue'
                img_gt_drawn = self._visualizer.add_datasample(
                    badcase_name if self.show else 'test_img',
                    img,
                    data_sample=data_sample,
                    show=False,
                    draw_pred=False,
                    draw_gt=True,
                    draw_bbox=False,
                    draw_heatmap=False,
                    wait_time=self.wait_time,
                    kpt_thr=self.kpt_thr,
                    out_file=None,
                    step=self._test_index)
                # draw pred keypoints in red color
                self._visualizer.kpt_color = 'red'
                self._visualizer.link_color = 'red'
                self._visualizer.add_datasample(
                    badcase_name if self.show else 'test_img',
                    img_gt_drawn,
                    data_sample=data_sample,
                    show=self.show,
                    draw_pred=True,
                    draw_gt=False,
                    draw_bbox=True,
                    draw_heatmap=False,
                    wait_time=self.wait_time,
                    kpt_thr=self.kpt_thr,
                    out_file=out_file,
                    step=self._test_index)

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        if not self.enable or not self.results:
            return

        mmengine.mkdir_or_exist(self.out_dir)
        out_file = os.path.join(self.out_dir, 'results.json')
        with open(out_file, 'w') as f:
            json.dump(self.results, f)

        print_log(
            f'the bad cases are saved under {self.out_dir}',
            logger='current',
            level=logging.INFO)
