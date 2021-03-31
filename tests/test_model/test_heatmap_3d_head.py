import numpy as np
import torch

from mmpose.models import HeatMap3DHead


def test_heatmap_3d_head():
    """Test interhand 3d head."""
    input_shape = (1, 512, 8, 8)
    inputs = torch.rand(input_shape, dtype=torch.float32)
    target_heatmap3d = inputs.new_zeros([1, 20, 64, 64, 64])
    target_weight = inputs.new_ones([1, 20, 1])
    img_metas = [{
        'img_shape': (224, 224, 3),
        'center': np.array([112, 112]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'bbox_id': 0,
        'flip_pairs': [],
        'inference_channel': np.arange(17),
        'image_file': '<demo>.png',
    }]

    # test 3D heatmap head
    head3d = HeatMap3DHead(
        in_channels=512,
        out_channels=20 * 64,
        depth_size=64,
        num_deconv_layers=3,
        num_deconv_filters=(256, 256, 256),
        num_deconv_kernels=(4, 4, 4),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    )
    head3d.init_weights()
    heatmap3d = head3d(inputs)
    assert heatmap3d.shape == torch.Size([1, 20, 64, 64, 64])

    loss_3d = head3d.get_loss(heatmap3d, target_heatmap3d, target_weight)
    assert 'heatmap_loss' in loss_3d

    # test inference model
    output = head3d.inference_model(inputs, [(0, 1)])
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 20, 64, 64, 64)

    # test decode
    result = head3d.decode(img_metas, output)
    assert 'preds' in result
