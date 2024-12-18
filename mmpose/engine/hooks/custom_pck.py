import torch
from mmengine.hooks import Hook
from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.structures import merge_data_samples, PoseDataSample

from mmpose.registry import HOOKS

@HOOKS.register_module()
class PCKAccuracyTrainHook(Hook):
    """A custom hook to calculate and log PCK accuracy during training.

    Args:
        interval (int): The interval (in iterations) at which to log PCK accuracy.
        thr (float): The threshold for PCK calculation.
    """

    def __init__(self, interval=10, thr=0.05):
        self.interval = interval
        self.thr = thr

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Calculates PCK accuracy after each training iteration at a given interval."""
        if batch_idx % self.interval == 0:
            model = runner.model

            # Forward pass to get predictions
            with torch.no_grad():
                preds = model(**data_batch)

            # Ground truth keypoints
            gts = data_batch['keypoints']  # Assuming keypoints are in data_batch

            # Ensure that predictions and ground truth are properly shaped
            if preds is None or gts is None:
                runner.logger.warning("Predictions or ground truth keypoints are missing.")
                return

            # Calculate the PCK accuracy
            acc, _ = keypoint_pck_accuracy(preds, gts, thr=self.thr)

            # Log the PCK accuracy
            runner.logger.info(f'Training PCK accuracy @ {self.thr}: {acc:.4f}')
