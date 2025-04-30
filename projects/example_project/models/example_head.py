from mmpose.models import HeatmapHead
from mmpose.registry import MODELS


# Register your head to the `MODELS`.
@MODELS.register_module()
class ExampleHead(HeatmapHead):
    """Implements an example head.

    Implement the model head just like a normal pytorch module.
    """

    def __init__(self, **kwargs) -> None:
        print('Initializing ExampleHead...')
        super().__init__(**kwargs)

    def forward(self, feats):
        """Forward the network. The input is multi scale feature maps and the
        output is the coordinates.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output coordinates or heatmaps.
        """
        return super().forward(feats)

    def predict(self, feats, batch_data_samples, test_cfg={}):
        """Predict results from outputs. The behaviour of head during testing
        should be defined in this function.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): A list of
                data samples for instances in a batch
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """
        return super().predict(feats, batch_data_samples, test_cfg)

    def loss(self, feats, batch_data_samples, train_cfg={}) -> dict:
        """Calculate losses from a batch of inputs and data samples. The
        behaviour of head during training should be defined in this function.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): A list of
                data samples for instances in a batch
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """

        return super().loss(feats, batch_data_samples, train_cfg)
