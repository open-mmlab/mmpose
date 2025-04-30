import copy
from abc import ABCMeta, abstractmethod

import json_tricks as json
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from mmpose.core.evaluation.top_down_eval import keypoint_pck_accuracy
from mmpose.datasets import DATASETS
from mmpose.datasets.pipelines import Compose


@DATASETS.register_module()
class FewShotBaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        self.image_info = {}
        self.ann_info = {}

        self.annotations_path = ann_file
        if not img_prefix.endswith('/'):
            img_prefix = img_prefix + '/'
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['flip_pairs'] = None

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.db = []
        self.num_shots = 1
        self.paired_samples = []
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def _select_kpt(self, obj, kpt_id):
        """Select kpt."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, pair in zip(preds, self.paired_samples):
            item = self.db[pair[-1]]
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])

            mask_query = ((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            mask_sample = ((np.array(
                self.db[pair[0]]['joints_3d_visible'])[:, 0]) > 0)
            for id_s in pair[:-1]:
                mask_sample = np.bitwise_and(
                    mask_sample,
                    ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))
            masks.append(np.bitwise_and(mask_query, mask_sample))

            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))

        if 'PCK' in metrics:
            pck_avg = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks,
                                                    threshold_bbox):
                _, pck, _ = keypoint_pck_accuracy(
                    np.expand_dims(output, 0), np.expand_dims(gt, 0),
                    np.expand_dims(mask, 0), pck_thr,
                    np.expand_dims(thr_bbox, 0))
                pck_avg.append(pck)
            info_str.append(('PCK', np.mean(pck_avg)))

        return info_str

    def _merge_obj(self, Xs_list, Xq, idx):
        """merge Xs_list and Xq.

        :param Xs_list: N-shot samples X
        :param Xq: query X
        :param idx: id of paired_samples
        :return: Xall
        """
        Xall = dict()
        Xall['img_s'] = [Xs['img'] for Xs in Xs_list]
        Xall['target_s'] = [Xs['target'] for Xs in Xs_list]
        Xall['target_weight_s'] = [Xs['target_weight'] for Xs in Xs_list]
        xs_img_metas = [Xs['img_metas'].data for Xs in Xs_list]

        Xall['img_q'] = Xq['img']
        Xall['target_q'] = Xq['target']
        Xall['target_weight_q'] = Xq['target_weight']
        xq_img_metas = Xq['img_metas'].data

        img_metas = dict()
        for key in xq_img_metas.keys():
            img_metas['sample_' + key] = [
                xs_img_meta[key] for xs_img_meta in xs_img_metas
            ]
            img_metas['query_' + key] = xq_img_metas[key]
        img_metas['bbox_id'] = idx

        Xall['img_metas'] = DC(img_metas, cpu_only=True)

        return Xall

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.paired_samples)

    def __getitem__(self, idx):
        """Get the sample given index."""

        pair_ids = self.paired_samples[idx]
        assert len(pair_ids) == self.num_shots + 1
        sample_id_list = pair_ids[:self.num_shots]
        query_id = pair_ids[-1]

        sample_obj_list = []
        for sample_id in sample_id_list:
            sample_obj = copy.deepcopy(self.db[sample_id])
            sample_obj['ann_info'] = copy.deepcopy(self.ann_info)
            sample_obj_list.append(sample_obj)

        query_obj = copy.deepcopy(self.db[query_id])
        query_obj['ann_info'] = copy.deepcopy(self.ann_info)

        if not self.test_mode:
            # randomly select "one" keypoint
            sample_valid = (sample_obj_list[0]['joints_3d_visible'][:, 0] > 0)
            for sample_obj in sample_obj_list:
                sample_valid = sample_valid & (
                    sample_obj['joints_3d_visible'][:, 0] > 0)
            query_valid = (query_obj['joints_3d_visible'][:, 0] > 0)

            valid_s = np.where(sample_valid)[0]
            valid_q = np.where(query_valid)[0]
            valid_sq = np.where(sample_valid & query_valid)[0]
            if len(valid_sq) > 0:
                kpt_id = np.random.choice(valid_sq)
            elif len(valid_s) > 0:
                kpt_id = np.random.choice(valid_s)
            elif len(valid_q) > 0:
                kpt_id = np.random.choice(valid_q)
            else:
                kpt_id = np.random.choice(np.array(range(len(query_valid))))

            for i in range(self.num_shots):
                sample_obj_list[i] = self._select_kpt(sample_obj_list[i],
                                                      kpt_id)
            query_obj = self._select_kpt(query_obj, kpt_id)

        # when test, all keypoints will be preserved.

        Xs_list = []
        for sample_obj in sample_obj_list:
            Xs = self.pipeline(sample_obj)
            Xs_list.append(Xs)
        Xq = self.pipeline(query_obj)

        Xall = self._merge_obj(Xs_list, Xq, idx)
        Xall['skeleton'] = self.db[query_id]['skeleton']

        return Xall

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
