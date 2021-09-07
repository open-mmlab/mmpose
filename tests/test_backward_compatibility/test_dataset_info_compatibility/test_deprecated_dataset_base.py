# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmpose.datasets.datasets.animal.animal_base_dataset import \
    AnimalBaseDataset
from mmpose.datasets.datasets.body3d.body3d_base_dataset import \
    Body3DBaseDataset
from mmpose.datasets.datasets.bottom_up.bottom_up_base_dataset import \
    BottomUpBaseDataset
from mmpose.datasets.datasets.face.face_base_dataset import FaceBaseDataset
from mmpose.datasets.datasets.fashion.fashion_base_dataset import \
    FashionBaseDataset
from mmpose.datasets.datasets.hand.hand_base_dataset import HandBaseDataset
from mmpose.datasets.datasets.top_down.topdown_base_dataset import \
    TopDownBaseDataset


@pytest.mark.parametrize('BaseDataset',
                         (AnimalBaseDataset, BottomUpBaseDataset,
                          FaceBaseDataset, FashionBaseDataset, HandBaseDataset,
                          TopDownBaseDataset, Body3DBaseDataset))
def test_dataset_base_class(BaseDataset):
    with pytest.raises(ImportError):

        class Dataset(BaseDataset):
            pass

        _ = Dataset()
