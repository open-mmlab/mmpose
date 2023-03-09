# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import numpy as np
import torch

from mmpose.models.utils import SMPL
from tests.utils.mesh_utils import generate_smpl_weight_file


def test_smpl():
    """Test smpl model."""

    # build smpl model
    smpl = None
    with tempfile.TemporaryDirectory() as tmpdir:
        # generate weight file for SMPL model.
        generate_smpl_weight_file(tmpdir)

        smpl_cfg = dict(
            smpl_path=tmpdir,
            joints_regressor=osp.join(tmpdir, 'test_joint_regressor.npy'))
        smpl = SMPL(**smpl_cfg)

    assert smpl is not None, 'Fail to build SMPL model'

    # test get face function
    faces = smpl.get_faces()
    assert isinstance(faces, np.ndarray)

    betas = torch.zeros(3, 10)
    body_pose = torch.zeros(3, 23 * 3)
    global_orient = torch.zeros(3, 3)
    transl = torch.zeros(3, 3)
    gender = torch.LongTensor([-1, 0, 1])

    # test forward with body_pose and global_orient in axis-angle format
    smpl_out = smpl(
        betas=betas, body_pose=body_pose, global_orient=global_orient)
    assert isinstance(smpl_out, dict)
    assert smpl_out['vertices'].shape == torch.Size([3, 6890, 3])
    assert smpl_out['joints'].shape == torch.Size([3, 24, 3])

    # test forward with body_pose and global_orient in rotation matrix format
    body_pose = torch.eye(3).repeat([3, 23, 1, 1])
    global_orient = torch.eye(3).repeat([3, 1, 1, 1])
    _ = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient)

    # test forward with translation
    _ = smpl(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl)

    # test forward with gender
    _ = smpl(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        gender=gender)

    # test forward when all samples in the same gender
    gender = torch.LongTensor([0, 0, 0])
    _ = smpl(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        gender=gender)

    # test forward when batch size = 0
    _ = smpl(
        betas=torch.zeros(0, 10),
        body_pose=torch.zeros(0, 23 * 3),
        global_orient=torch.zeros(0, 3))
