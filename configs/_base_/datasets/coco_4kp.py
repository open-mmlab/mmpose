# Scaletronic 4kp
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='',
        title='Label: 4 keypoints dataset ',
        container='',
        year='2025',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='front_right', 
            id=0, 
            color=[51, 153, 255], 
            type='upper', 
            swap='front_left'
        ),
        1:
        dict(
            name='rear_right',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='rear_left'),
        2:
        dict(
            name='front_left',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='front_right'),
        3:
        dict(
            name='rear_left',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='rear_right')
    },
    skeleton_info={
        0:
        dict(link=('front_left', 'front_right'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('rear_left', 'rear_right'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('front_left', 'rear_left'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('front_right', 'rear_right'), id=3, color=[0, 255, 0])
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[
        0.05, 0.05, 0.05, 0.05
    ])