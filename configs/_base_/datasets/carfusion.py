dataset_info = dict(
    dataset_name='carfusion',
    paper_info=dict(
        author='N Dinesh Reddy and Minh Vo and '
        'Srinivasa Narasimhan',
        title='CarFusion:Combining Point Tracking'
        'and Part Detection for Dynamic 3D Reconstruction of Vehicle',
        container='Computer Vision and Patter Recognition',
        year='2018',
        homepage='https://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion'
        '/cvpr2018/index.html',
    ),
    keypoint_info={
        0:
        dict(
            name='Right Front Wheel',
            id=0,
            color=[51, 153, 255],
            type='front',
            swap='Left Front Wheel'),
        1:
        dict(
            name='Left Front Wheel',
            id=1,
            color=[51, 153, 255],
            type='front',
            swap='Right Front Wheel'),
        2:
        dict(
            name='Right Back Wheel',
            id=2,
            color=[51, 153, 255],
            type='back',
            swap='Left Back Wheel'),
        3:
        dict(
            name='Left Back Wheel',
            id=3,
            color=[51, 153, 255],
            type='back',
            swap='Right Back Wheel'),
        4:
        dict(
            name='Right Front HeadLight',
            id=4,
            color=[51, 153, 255],
            type='front',
            swap='Left Front HeadLight'),
        5:
        dict(
            name='Left Front HeadLight',
            id=5,
            color=[0, 255, 0],
            type='front',
            swap='Right Front HeadLight'),
        6:
        dict(
            name='Right Back HeadLight',
            id=6,
            color=[255, 128, 0],
            type='back',
            swap='Left Back HeadLight'),
        7:
        dict(
            name='Left Back HeadLight',
            id=7,
            color=[0, 255, 0],
            type='back',
            swap='Right Back HeadLight'),
        8:
        dict(name='Exhaust', id=8, color=[255, 128, 0], type='back', swap=''),
        9:
        dict(
            name='Right Front Top',
            id=9,
            color=[0, 255, 0],
            type='front',
            swap='Left Front Top'),
        10:
        dict(
            name='Left Front Top',
            id=10,
            color=[0, 255, 0],
            type='front',
            swap='Right Front Top'),
        11:
        dict(
            name='misc',
            id=11,
            color=[255, 255, 255],
            type='back',
            swap='misc'),
        12:
        dict(
            name='Left Back Top',
            id=12,
            color=[255, 0, 0],
            type='back',
            swap='Right Back Top'),
        13:
        dict(
            name='Right Back Top',
            id=13,
            color=[0, 255, 0],
            type='back',
            swap='Left Back Top'),
    },
    skeleton_info={
        0:
        dict(
            link=('Right Front Wheel', 'Left Front Wheel'),
            id=0,
            color=[0, 255, 0]),
        1:
        dict(
            link=('Right Front Wheel', 'Right Back Wheel'),
            id=1,
            color=[0, 255, 0]),
        2:
        dict(
            link=('Right Front Wheel', 'Right Front HeadLight'),
            id=2,
            color=[255, 128, 0]),
        3:
        dict(
            link=('Left Front Wheel', 'Left Back Wheel'),
            id=3,
            color=[255, 128, 0]),
        4:
        dict(
            link=('Left Front Wheel', 'Left Front HeadLight'),
            id=4,
            color=[51, 153, 255]),
        5:
        dict(
            link=('Left Front HeadLight', 'Right Front HeadLight'),
            id=5,
            color=[0, 255, 0]),
        6:
        dict(
            link=('Right Back HeadLight', 'Left Back HeadLight'),
            id=6,
            color=[255, 128, 0]),
        7:
        dict(
            link=('Left Back HeadLight', 'Left Back Top'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(
            link=('Left Back HeadLight', 'Left Back Wheel'),
            id=8,
            color=[51, 153, 255]),
        9:
        dict(
            link=('Right Back HeadLight', 'Right Back Wheel'),
            id=9,
            color=[51, 153, 255]),
        10:
        dict(
            link=('Right Back HeadLight', 'Right Back Top'),
            id=10,
            color=[51, 153, 255]),
        11:
        dict(
            link=('Right Back Wheel', 'Left Back Wheel'),
            id=11,
            color=[51, 153, 255]),
        12:
        dict(
            link=('Right Front Top', 'Right Back Top'),
            id=12,
            color=[51, 153, 255]),
        13:
        dict(
            link=('Right Front Top', 'Right Front HeadLight'),
            id=13,
            color=[51, 153, 255]),
        14:
        dict(
            link=('Left Front Top', 'Left Back Top'),
            id=14,
            color=[51, 153, 255]),
        15:
        dict(
            link=('Left Front Top', 'Left Front HeadLight'),
            id=15,
            color=[0, 255, 0]),
        16:
        dict(
            link=('Left Front Top', 'Right Front Top'),
            id=16,
            color=[255, 128, 0]),
        17:
        dict(
            link=('Left Back Top', 'Right Back Top'),
            id=17,
            color=[255, 128, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
