# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class AnimalKingdomDataset(BaseCocoStyleDataset):
    """Animal Kingdom dataset for animal pose estimation.

    "[CVPR2022] Animal Kingdom:
     A Large and Diverse Dataset for Animal Behavior Understanding"
    More details can be found in the `paper
    <https://www.researchgate.net/publication/
    359816954_Animal_Kingdom_A_Large_and_Diverse
    _Dataset_for_Animal_Behavior_Understanding>`__ .

    Website: <https://sutdcv.github.io/Animal-Kingdom>

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Animal Kingdom keypoint indexes::

        0: 'Head_Mid_Top',
        1: 'Eye_Left',
        2: 'Eye_Right',
        3: 'Mouth_Front_Top',
        4: 'Mouth_Back_Left',
        5: 'Mouth_Back_Right',
        6: 'Mouth_Front_Bottom',
        7: 'Shoulder_Left',
        8: 'Shoulder_Right',
        9: 'Elbow_Left',
        10: 'Elbow_Right',
        11: 'Wrist_Left',
        12: 'Wrist_Right',
        13: 'Torso_Mid_Back',
        14: 'Hip_Left',
        15: 'Hip_Right',
        16: 'Knee_Left',
        17: 'Knee_Right',
        18: 'Ankle_Left ',
        19: 'Ankle_Right',
        20: 'Tail_Top_Back',
        21: 'Tail_Mid_Back',
        22: 'Tail_End_Back
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/ak.py')
