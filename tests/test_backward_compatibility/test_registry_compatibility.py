# Copyright (c) OpenMMLab. All rights reserved.
import pytest


def test_old_fashion_registry_importing():
    with pytest.warns(DeprecationWarning):
        from mmpose.models.registry import (  # noqa: F401
            BACKBONES, NECKS, HEADS, LOSSES, POSENETS)
    with pytest.warns(DeprecationWarning):
        from mmpose.datasets.registry import DATASETS, PIPELINES  # noqa: F401
