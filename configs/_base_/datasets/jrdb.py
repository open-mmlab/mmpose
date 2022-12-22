dataset_info = dict(
    dataset_name='jrdb',
    paper_info=dict(
        author='',
        title='JRDB-Pose',
        container='European conference on computer vision',
        year='2022',
        homepage='https://jrdb.erc.monash.edu/',
    ),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='right eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='left eye'),
        2:
        dict(
            name='left eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='right eye'),
        3:
        dict(
            name='right shoulder',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left shoulder'),
        4:
        dict(
            name='center shoulder', 
            id=4, 
            color=[51, 153, 255], 
            type='upper', 
            swap=''),
        5:
        dict(
            name='left shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right shoulder'),
        6:
        dict(
            name='right elbow',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left elbow'),
        7:
        dict(
            name='left elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right elbow'),
        8:
        dict(
            name='center hip', 
            id=8, 
            color=[51, 153, 255], 
            type='lower', 
            swap=''),
        9:
        dict(
            name='right wrist',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='left wrist'),
        10:
        dict(
            name='right hip',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='left hip'),
        11:
        dict(
            name='left hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right hip'),
        12:
        dict(
            name='left wrist',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='right wrist'),
        13:
        dict(
            name='right knee',
            id=13,
            color=[255, 128, 0],
            type='lower',
            swap='left knee'),
        14:
        dict(
            name='left knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='right knee'),
        15:
        dict(
            name='right foot',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='left foot'),
        16:
        dict(
            name='left foot',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='right foot')
    },
    skeleton_info={
        0:
        dict(link=('left foot', 'left knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left knee', 'left hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right foot', 'right knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right knee', 'right hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left hip', 'right hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left shoulder', 'left hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right shoulder', 'right hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left shoulder', 'right shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left shoulder', 'left elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right shoulder', 'right elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left elbow', 'left wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right elbow', 'right wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left eye', 'right eye'), id=12, color=[51, 153, 255]),
        13:
        dict(
            link=('center hip', 'right hip'),
            id=13,
            color=[51, 153, 255]),
        14:
        dict(
            link=('center hip', 'left hip'),
            id=14,
            color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    # Adapted from COCO dataset
    sigmas=[
        0.079, 0.025, 0.025, 0.079, 0.026, 0.079, 0.072, 0.072, 0.107, 0.062, 0.107, 0.107, 0.062, 0.087, 0.087, 0.089, 0.089
    ])
