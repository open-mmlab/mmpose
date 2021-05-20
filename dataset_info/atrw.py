dataset_info = dict(
    paper_info=dict(
        author='Li, Shuyuan and Li, Jianguo and Tang, Hanlin'
        ' and Qian, Rui and Lin, Weiyao',
        title='ATRW: A Benchmark for Amur Tiger'
        ' Re-identification in the Wild',
        booktitle='Proceedings of the 28th ACM '
        'International Conference on Multimedia',
        year='2020'),
    dataset_name='atrw',
    keypoint_name=[
        'left_ear', 'right_ear', 'nose', 'right_shoulder', 'right_front_paw',
        'left_shoulder', 'left_front_paw', 'right_hip', 'right_knee',
        'right_back_paw', 'left_hip', 'left_knee', 'left_back_paw', 'tail',
        'center'
    ],
    skeleton=[[0, 2], [1, 2], [2, 14], [5, 6], [5, 14], [3, 4], [3, 14],
              [13, 14], [9, 8], [8, 7], [7, 13], [12, 11], [11, 10], [10, 13]],
    flip_pairs=[[0, 1], [3, 5], [4, 6], [7, 10], [8, 11], [9, 12]],
    upper_body_ids=(0, 1, 2, 3, 4, 5, 6),
    lower_body_ids=(7, 8, 9, 10, 11, 12, 13, 14),
    joint_weights=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],

    # `ATRW: A Benchmark for Amur Tiger Re-identification in the Wild'
    sigmas=[
        0.0277, 0.0823, 0.0831, 0.0202, 0.0716, 0.0263, 0.0646, 0.0302, 0.0440,
        0.0316, 0.0333, 0.0547, 0.0263, 0.0683, 0.0539
    ])
