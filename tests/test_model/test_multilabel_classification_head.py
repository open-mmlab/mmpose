import numpy as np
import torch

from mmpose.models import MultilabelClassificationHead


def test_multilabel_classification_head():
    """Test multi-label classification head."""
    inputs = torch.rand([1, 512], dtype=torch.float32)
    target = inputs.new_zeros([1, 2])
    target_weight = inputs.new_ones([1, 2])
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

    # build multi-label classification head
    head = MultilabelClassificationHead(
        in_channels=512,
        num_labels=2,
        hidden_dims=(256, ),
        loss_classification=dict(type='BCELoss', use_target_weight=True))
    head.init_weights()

    # test forward function
    labels = head(inputs)
    assert labels.shape == torch.Size([1, 2])
    loss = head.get_loss(labels, target, target_weight)
    assert 'classification_loss' in loss
    acc = head.get_accuracy(labels, target, target_weight)
    assert 'acc_classification' in acc

    acc = head.get_accuracy(
        inputs.new_ones(1, 2), inputs.new_ones(1, 2), inputs.new_ones(1, 2))
    assert torch.allclose(acc['acc_classification'], torch.tensor(1.))

    acc = head.get_accuracy(
        inputs.new_zeros(1, 2), inputs.new_ones(1, 2), inputs.new_ones(1, 2))
    assert torch.allclose(acc['acc_classification'], torch.tensor(0.))

    # test inference model function
    output = head.inference_model(inputs, [(0, 1)])
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)

    # test decode function
    result = head.decode(img_metas, output)
    assert 'labels' in result
