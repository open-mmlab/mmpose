import pytest
import torch

from mmpose.models import OpenPoseNetworkV2


def test_openpose_network_v2_backbone():
    with pytest.raises(AssertionError):
        # OpenPoseNetwork's num_stacks should larger than 0
        OpenPoseNetworkV2(in_channels=3, num_stages=-1)

    with pytest.raises(AssertionError):
        # OpenPoseNetwork's in_channels should be 3
        OpenPoseNetworkV2(in_channels=2)

    with pytest.raises(AssertionError):
        # len(stage_types) == num_stages
        OpenPoseNetworkV2(
            in_channels=3, num_stages=3, stage_types=('PAF', 'CM'))

    with pytest.raises(ValueError):
        # stage_type should be either 'CM' or 'PAF'.
        OpenPoseNetworkV2(
            in_channels=3, num_stages=2, stage_types=('PAF', 'CC'))

    # Test OpenPoseNetwork
    model = OpenPoseNetworkV2(in_channels=3)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 368, 368)
    feat = model(imgs)
    assert len(feat) == 6
    assert feat[0].shape == torch.Size([1, 38, 46, 46])
    assert feat[-1].shape == torch.Size([1, 19, 46, 46])


test_openpose_network_v2_backbone()
