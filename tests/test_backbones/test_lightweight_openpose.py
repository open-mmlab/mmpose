import pytest
import torch

from mmpose.models import LightweightOpenPoseNetwork


def test_lightweight_openpose_network_backbone():
    with pytest.raises(AssertionError):
        # OpenPoseNetwork's num_stacks should larger than 0
        LightweightOpenPoseNetwork(in_channels=3, num_stages=-1)

    with pytest.raises(AssertionError):
        # OpenPoseNetwork's in_channels should be 3
        LightweightOpenPoseNetwork(in_channels=2)

    # Test OpenPoseNetwork
    model = LightweightOpenPoseNetwork(in_channels=3)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 368, 368)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 19, 46, 46])
    assert feat[-1].shape == torch.Size([1, 38, 46, 46])
