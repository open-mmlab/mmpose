import numpy as np
import torch

from mmpose.models import Heatmap1DHead


def test_heatmap_1d_head():
    """Test heatmap 1d head."""
    inputs = torch.rand([1, 512], dtype=torch.float32)
    target = inputs.new_zeros([1, 1])
    target_weight = inputs.new_ones([1, 1])
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

    # build 1D heatmap head
    head = Heatmap1DHead(
        in_channels=512,
        heatmap_size=64,
        hidden_dims=(512, ),
        loss_value=dict(type='L1Loss'))
    head.init_weights()
    # test forward function
    value = head(inputs)
    assert value.shape == torch.Size([1, 1])

    loss_value = head.get_loss(value, target, target_weight)
    assert 'value_loss' in loss_value

    # test inference model function
    output = head.inference_model(inputs)
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 1)

    # test decode function
    result = head.decode(img_metas, output)
    assert 'values' in result
