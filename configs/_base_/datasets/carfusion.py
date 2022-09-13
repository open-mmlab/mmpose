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
            name='Right Back Top',
            id=11,
            color=[255, 0, 0],
            type='back',
            swap='Left Back Top'),
        12:
        dict(
            name='Left Back Top',
            id=12,
            color=[255, 0, 0],
            type='back',
            swap='Right Back Top'),
        13:
        dict(
            name='misc',
            id=13,
            color=[255, 255, 255],
            type='back',
            swap='misc'),

    },

    # keypoint_info={
    #     0:
    #         dict(
    #             name='Right Front Wheel',
    #             id=0,
    #             color=[51, 153, 255],
    #             type='front',
    #             swap='Left Front Wheel'),
    #     1:
    #         dict(
    #             name='Left Front Wheel',
    #             id=1,
    #             color=[51, 153, 255],
    #             type='front',
    #             swap='Right Front Wheel'),
    #     2:
    #         dict(
    #             name='Right Back Wheel',
    #             id=2,
    #             color=[51, 153, 255],
    #             type='back',
    #             swap='Left Back Wheel'),
    #     3:
    #         dict(
    #             name='Left Back Wheel',
    #             id=3,
    #             color=[51, 153, 255],
    #             type='back',
    #             swap='Right Back Wheel'),
    #     4:
    #         dict(
    #             name='Right Front HeadLight',
    #             id=4,
    #             color=[51, 153, 255],
    #             type='front',
    #             swap='Left Front HeadLight'),
    #     5:
    #         dict(
    #             name='Left Front HeadLight',
    #             id=5,
    #             color=[0, 255, 0],
    #             type='front',
    #             swap='Right Front HeadLight'),
    #     6:
    #         dict(
    #             name='Right Back HeadLight',
    #             id=6,
    #             color=[255, 128, 0],
    #             type='back',
    #             swap='Left Back HeadLight'),
    #     7:
    #         dict(
    #             name='Left Back HeadLight',
    #             id=7,
    #             color=[0, 255, 0],
    #             type='back',
    #             swap='Right Back HeadLight'),
    #     8:
    #         dict(name='Exhaust', id=8, color=[255, 128, 0], type='back', swap=''),
    #     9:
    #         dict(
    #             name='Right Front Top',
    #             id=9,
    #             color=[0, 255, 0],
    #             type='front',
    #             swap='Left Front Top'),
    #     10:
    #         dict(
    #             name='Left Front Top',
    #             id=10,
    #             color=[0, 255, 0],
    #             type='front',
    #             swap='Right Front Top'),
    #     11:
    #         dict(
    #             name='misc',
    #             id=11,
    #             color=[255, 255, 255],
    #             type='back',
    #             swap='misc'),
    #     12:
    #         dict(
    #             name='Left Back Top',
    #             id=12,
    #             color=[255, 0, 0],
    #             type='back',
    #             swap='Right Back Top'),
    #     13:
    #         dict(
    #             name='Right Back Top',
    #             id=13,
    #             color=[255, 0, 0],
    #             type='back',
    #             swap='Left Back Top'),
    # },

    skeleton_info={
        0:
        dict(
            link=('Right Front Wheel', 'Left Front Wheel'),
            id=0,
            color=[251.7364690861601, 80.29174516784451, 75.5]),
        1:
        dict(
            link=('Right Front Wheel', 'Right Back Wheel'),
            id=1,
            color=[255.0, 0.0, 127.5]),
        2:
        dict(
            link=('Right Front Wheel', 'Right Front HeadLight'),
            id=2,
            color=[201.58576513654165, 246.91039043351853, 86.5]),
        3:
        dict(
            link=('Left Front Wheel', 'Left Back Wheel'),
            id=3,
            color=[254.18280785388578, 40.666326758545104, 101.50000000000001]),
        4:
        dict(
            link=('Left Front Wheel', 'Left Front HeadLight'),
            id=4,
            color=[156.1671517775588, 246.91039043351853, 168.49999999999997]),
        5:
        dict(
            link=('Left Front HeadLight', 'Right Front HeadLight'),
            id=5,
            color=[172.94995492405468, 254.18280785388578, 140.5]),
        6:
        dict(
            link=('Right Back HeadLight', 'Left Back HeadLight'),
            id=6,
            color=[40.66632675854512, 80.29174516784454, 255.0]),
        7:
        dict(
            link=('Left Back HeadLight', 'Left Back Top'),
            id=7,
            color=[62.19799862763618, 120.63885699318259, 255.0]),
        8:
        dict(
            link=('Left Back HeadLight', 'Left Back Wheel'),
            id=8,
            color=[81.78111007551507, 154.9224313163983, 255.0]),
        9:
        dict(
            link=('Right Back HeadLight', 'Right Back Wheel'),
            id=9,
            color=[120.63885699318257, 212.56860823138547, 222.5]),
        10:
        dict(
            link=('Right Back HeadLight', 'Right Back Top'),
            id=10,
            color=[102.28089140784328, 187.3854665969788, 248.50000000000003]),
        11:
        dict(
            link=('Right Back Wheel', 'Left Back Wheel'),
            id=11,
            color=[247.29821868892742, 120.63885699318257, 47.5]),
        12:
        dict(
            link=('Right Front Top', 'Right Back Top'),
            id=12,
            color=[241.53022592383027, 154.92243131639827, 21.500000000000007]),
        13:
        dict(
            link=('Right Front Top', 'Right Front HeadLight'),
            id=13,
            color=[187.38546659697883, 254.18280785388578, 114.49999999999999]),
        14:
        dict(
            link=('Left Front Top', 'Left Back Top'),
            id=14,
            color=[233.58856832648505, 187.3854665969788, 6.500000000000005]),
        15:
        dict(
            link=('Left Front Top', 'Left Front HeadLight'),
            id=15,
            color=[139.54098446513834, 233.58856832648505, 194.5]),
        16:
        dict(
            link=('Left Front Top', 'Right Front Top'),
            id=16,
            color=[224.65810954287505, 212.56860823138544, 32.5]),
        17:
        dict(
            link=('Left Back Top', 'Right Back Top'),
            id=17,
            color=[213.43222262465434, 233.58856832648505, 60.500000000000014])
    },
    joint_weights=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    sigmas=[
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
        0.025, 0.025, 0.025, 0.025
    ])
