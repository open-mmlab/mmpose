dataset_info = dict(
    dataset_name='exlpose',
    paper_info=dict(
        author='Sohyun Lee, Jaesung Rim, Boseung Jeong, Geonu Kim,'
        'ByungJu Woo, Haechan Lee, Sunghyun Cho, Suha Kwak',
        title='Human Pose Estimation in Extremely Low-Light Conditions',
        container='arXiv',
        year='2023',
        homepage='https://arxiv.org/abs/2303.15410',
    ),
    keypoint_info={
        0:
        dict(
            name='left_shoulder',
            id=0,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        1:
        dict(
            name='right_shoulder',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        2:
        dict(
            name='left_elbow',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='left_wrist',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        5:
        dict(
            name='right_wrist',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        7:
        dict(
            name='right_hip',
            id=7,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        8:
        dict(
            name='left_knee',
            id=8,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        10:
        dict(
            name='left_ankle',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        12:
        dict(name='head', id=12, color=[51, 153, 255], type='upper', swap=''),
        13:
        dict(name='neck', id=13, color=[51, 153, 255], type='upper', swap='')
    },
    skeleton_info={
        0: dict(link=('head', 'neck'), id=0, color=[51, 153, 255]),
        1: dict(link=('neck', 'left_shoulder'), id=1, color=[51, 153, 255]),
        2: dict(link=('neck', 'right_shoulder'), id=2, color=[51, 153, 255]),
        3: dict(link=('left_shoulder', 'left_elbow'), id=3, color=[0, 255, 0]),
        4: dict(link=('left_elbow', 'left_wrist'), id=4, color=[0, 255, 0]),
        5: dict(
            link=('right_shoulder', 'right_elbow'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('right_elbow', 'right_wrist'), id=6, color=[255, 128, 0]),
        7: dict(link=('neck', 'right_hip'), id=7, color=[51, 153, 255]),
        8: dict(link=('neck', 'left_hip'), id=8, color=[51, 153, 255]),
        9: dict(link=('right_hip', 'right_knee'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('right_knee', 'right_ankle'), id=10, color=[255, 128, 0]),
        11: dict(link=('left_hip', 'left_knee'), id=11, color=[0, 255, 0]),
        12: dict(link=('left_knee', 'left_ankle'), id=12, color=[0, 255, 0]),
    },
    joint_weights=[
        0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2, 0.2, 0.5
    ],
    sigmas=[
        0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089, 0.079, 0.079
    ])
