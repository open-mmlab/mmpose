# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle

import numpy as np
from scipy.sparse import csc_matrix


def generate_smpl_weight_file(output_dir):
    """Generate a SMPL model weight file to initialize SMPL model, and generate
    a 3D joints regressor file."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joint_regressor_file = os.path.join(output_dir, 'test_joint_regressor.npy')
    np.save(joint_regressor_file, np.zeros([24, 6890]))

    test_data = {}
    test_data['f'] = np.zeros([1, 3], dtype=np.int32)
    test_data['J_regressor'] = csc_matrix(np.zeros([24, 6890]))
    test_data['kintree_table'] = np.zeros([2, 24], dtype=np.uint32)
    test_data['J'] = np.zeros([24, 3])
    test_data['weights'] = np.zeros([6890, 24])
    test_data['posedirs'] = np.zeros([6890, 3, 207])
    test_data['v_template'] = np.zeros([6890, 3])
    test_data['shapedirs'] = np.zeros([6890, 3, 10])

    with open(os.path.join(output_dir, 'SMPL_NEUTRAL.pkl'), 'wb') as out_file:
        pickle.dump(test_data, out_file)
    with open(os.path.join(output_dir, 'SMPL_MALE.pkl'), 'wb') as out_file:
        pickle.dump(test_data, out_file)
    with open(os.path.join(output_dir, 'SMPL_FEMALE.pkl'), 'wb') as out_file:
        pickle.dump(test_data, out_file)
    return
