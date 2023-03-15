from mmpose.models import KeypointMSELoss
from mmpose.registry import MODELS


# Register your loss to the `MODELS`.
@MODELS.register_module()
class ExampleLoss(KeypointMSELoss):
    """Implements an example loss.

    Implement the loss just like a normal pytorch module.
    """

    def __init__(self, **kwargs) -> None:
        print('Initializing ExampleLoss...')
        super().__init__(**kwargs)

    def forward(self, output, target, target_weights=None, mask=None):
        """Forward function of loss. The input arguments should match those
        given in `head.loss` function.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        return super().forward(output, target, target_weights, mask)
