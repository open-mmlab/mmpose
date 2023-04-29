dataset_info = dict(
    dataset_name='linemod',
    paper_info=dict(
        author='',
        title='',
        container='',
        year='',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(name='min_min_min',
                id=0,
                color=[0,0,0],
                type='',
                swap=''),
        1:
        dict(name='min_min_max',
                id=1,
                color=[0,0,0],
                type='',
                swap=''),
        2:
        dict(name='min_max_min',
                id=2,
                color=[0,0,0],
                type='',
                swap=''),
        3:
        dict(name='min_max_max',
                id=3,
                color=[0,0,0],
                type='',
                swap=''),
        4:
        dict(name='max_min_min',
                id=4,
                color=[0,0,0],
                type='',
                swap=''),
        5:
        dict(name='max_min_max',
                id=5,
                color=[0,0,0],
                type='',
                swap=''),
        6:
        dict(name='max_max_min',
                id=6,
                color=[0,0,0],
                type='',
                swap=''),
        7:
        dict(name='max_max_max',
                id=7,
                color=[0,0,0],
                type='',
                swap=''),
    },
    skeleton_info={
        0:
        dict(link=('min_min_min', 'max_min_min'), id=0, color=[255,0,0]),
        1:
        dict(link=('min_min_max', 'max_min_max'), id=1, color=[255,0,0]),
        2:
        dict(link=('min_max_max', 'max_max_max'), id=2, color=[255,0,0]),
        3:
        dict(link=('max_max_min', 'min_max_min'), id=3, color=[255,0,0]),
        4:
        dict(link=('min_min_min', 'min_max_min'), id=4, color=[0,255,0]),
        5:
        dict(link=('min_min_max', 'min_max_max'), id=5, color=[0,255,0]),
        6:
        dict(link=('max_max_max', 'max_min_max'), id=6, color=[0,255,0]),
        7:
        dict(link=('max_min_min', 'max_max_min'), id=7, color=[0,255,0]),
        8:
        dict(link=('min_min_min', 'min_min_max'), id=8, color=[0,0,255]),
        9:
        dict(link=('max_max_max', 'max_max_min'), id=9, color=[0,0,255]),
        10:
        dict(link=('max_min_max', 'max_min_min'), id=10, color=[0,0,255]),
        11:
        dict(link=('min_max_min', 'min_max_max'), id=11, color=[0,0,255])
    },
    joint_weights=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    sigmas=[0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
            0.025, 0.025]   
)