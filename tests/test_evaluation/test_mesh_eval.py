# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.core import compute_similarity_transform


def test_compute_similarity_transform():
    source = np.random.rand(14, 3)
    tran = np.random.rand(1, 3)
    scale = 0.5
    target = source * scale + tran
    source_transformed = compute_similarity_transform(source, target)
    assert_array_almost_equal(source_transformed, target)
