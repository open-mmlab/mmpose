dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='',
        title='Forklift: 9 keypoints',
        container='',
        year='2025',
        homepage='',
    ),
    keypoint_info = {
        0: dict(
            name='C_Fork',
            id=0,
            color=[255, 128, 0],
            type='upper',
            swap=''
        ),
        1: dict(
            name='L_Fork',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='R_Fork'
        ),
        2: dict(
            name='R_Fork',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='L_Fork'
        ),
        3: dict(
            name='front_left',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='front_right'
        ),
        4: dict(
            name='front_right',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='front_left'
        ),
        5: dict(
            name='rear_left',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='rear_right'
        ),
        6: dict(
            name='rear_right',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap='rear_left'
        ),
        7: dict(
            name='tip_left',
            id=7,
            color=[0, 153, 255],
            type='upper',
            swap='tip_right'
        ),
        8: dict(
            name='tip_right',
            id=8,
            color=[51, 153, 0],
            type='upper',
            swap='tip_left'
        ),
    },
    skeleton_info={
        0:
        dict(link=('rear_left', 'rear_right'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('front_left', 'front_right'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('rear_left', 'front_left'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('rear_right', 'front_right'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('L_Fork', 'R_Fork'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('L_Fork', 'C_Fork'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('C_Fork', 'R_Fork'), id=6, color=[255, 128, 0]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1.,1.,1.,
    ],
    sigmas=[
        0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05,0.07,0.07
    ])
