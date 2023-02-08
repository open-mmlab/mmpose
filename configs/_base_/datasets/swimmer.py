dataset_info = dict(
    dataset_name='swimmer',
    keypoint_info={
        0:
        dict(
            name='Neck',
            id=0,
            color=[51, 153, 255],
            type='upper'),
        1:
        dict(
            name='LShoulder',
            id=1,
            color=[0, 255, 0],
            type='upper',
            swap='RShoulder'),
        2:
        dict(
            name='LElbow',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='RElbow'),
        3:
        dict(
            name='LWrist',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='RWrist'),
        4:
        dict(
            name='RShoulder',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='LShoulder'),
        5:
        dict(
            name='RElbow',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap='LElbow'),
        6:
        dict(
            name='RWrist',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='LWrist')
    },
    skeleton_info={
        0:
        dict(link=('Neck', 'LShoulder'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('Neck', 'RShoulder'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('LShoulder', 'LElbow'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('LElbow', 'LWrist'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('RShoulder', 'RElbow'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('RElbow', 'RWrist'), id=5, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[0.026, 0.079, 0.072, 0.062, 0.079, 0.072, 0.062])
