from ..builder import POSENETS
from .top_down import TopDown


@POSENETS.register_module()
class Interhand3D(TopDown):
    """Top-down interhand 3D pose detector of paper ref: Gyeongsik Moon.

    "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image". A child class of TopDown detector.
    """

    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (list[torch.Tensor]): Target heatmaps, relative hand
            root depth and hand type.
            target_weight (list[torch.Tensor]): Weights for target
            heatmaps, relative hand root depth and hand type.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint 3D
                heatmap
                - "root_depth_bound": depth bound of relative root depth 1D
                heatmap
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes, image paths, heatmaps,
              relative hand root depth and hand type.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(img, img_metas, **kwargs)

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output = [(out + out_flipped) * 0.5
                          for out, out_flipped in zip(output, output_flipped)]

        if self.with_keypoint:
            result = self.keypoint_head.decode(
                img_metas, output, img_size=[img_width, img_height])
        else:
            result = {}
        return result
