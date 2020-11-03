from .datasets.builder import DATASETS
from .datasets.datasets.top_down.topdown_base_dataset import TopDownBaseDataset


@DATASETS.register_module()
class TopDownFreiHandDataset(TopDownBaseDataset):
    """Deprecated TopDownFreiHandDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'TopDownFreiHandDataset has been renamed into FreiHandDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/202 for details'))

    def _get_db(self):
        return []

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return None


@DATASETS.register_module()
class TopDownOneHand10KDataset(TopDownBaseDataset):
    """Deprecated TopDownOneHand10KDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'TopDownOneHand10KDataset has been renamed into OneHand10KDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/202 for details'))

    def _get_db(self):
        return []

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return None


@DATASETS.register_module()
class TopDownPanopticDataset(TopDownBaseDataset):
    """Deprecated TopDownPanopticDataset."""

    def __init__(self, *args, **kwargs):
        raise (ImportError(
            'TopDownPanopticDataset has been renamed into PanopticDataset,'
            'check https://github.com/open-mmlab/mmpose/pull/202 for details'))

    def _get_db(self):
        return []

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return None
