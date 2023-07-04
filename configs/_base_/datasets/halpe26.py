dataset_info = dict(
    dataset_name='halpe26',
    paper_info=dict(
        author='Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie'
        ' and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu'
        ' and Ma, Ze and Chen, Mingyang and Lu, Cewu',
        title='PaStaNet: Toward Human Activity Knowledge Engine',
        container='CVPR',
        year='2020',
        homepage='https://github.com/Fang-Haoshu/Halpe-FullBody/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        17:
        dict(name='head', id=17, color=[255, 128, 0], type='upper', swap=''),
        18:
        dict(name='neck', id=18, color=[255, 128, 0], type='upper', swap=''),
        19:
        dict(name='hip', id=19, color=[255, 128, 0], type='lower', swap=''),
        20:
        dict(
            name='left_big_toe',
            id=20,
            color=[255, 128, 0],
            type='lower',
            swap='right_big_toe'),
        21:
        dict(
            name='right_big_toe',
            id=21,
            color=[255, 128, 0],
            type='lower',
            swap='left_big_toe'),
        22:
        dict(
            name='left_small_toe',
            id=22,
            color=[255, 128, 0],
            type='lower',
            swap='right_small_toe'),
        23:
        dict(
            name='right_small_toe',
            id=23,
            color=[255, 128, 0],
            type='lower',
            swap='left_small_toe'),
        24:
        dict(
            name='left_heel',
            id=24,
            color=[255, 128, 0],
            type='lower',
            swap='right_heel'),
        25:
        dict(
            name='right_heel',
            id=25,
            color=[255, 128, 0],
            type='lower',
            swap='left_heel')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_hip', 'hip'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('right_ankle', 'right_knee'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('right_knee', 'right_hip'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('right_hip', 'hip'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('head', 'neck'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('neck', 'hip'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('neck', 'left_shoulder'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('left_shoulder', 'left_elbow'), id=9, color=[0, 255, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('neck', 'right_shoulder'), id=11, color=[255, 128, 0]),
        12:
        dict(
            link=('right_shoulder', 'right_elbow'), id=12, color=[255, 128,
                                                                  0]),
        13:
        dict(link=('right_elbow', 'right_wrist'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('left_eye', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('nose', 'left_eye'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('nose', 'right_eye'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_eye', 'left_ear'), id=17, color=[51, 153, 255]),
        18:
        dict(link=('right_eye', 'right_ear'), id=18, color=[51, 153, 255]),
        19:
        dict(link=('left_ear', 'left_shoulder'), id=19, color=[51, 153, 255]),
        20:
        dict(
            link=('right_ear', 'right_shoulder'), id=20, color=[51, 153, 255]),
        21:
        dict(link=('left_ankle', 'left_big_toe'), id=21, color=[0, 255, 0]),
        22:
        dict(link=('left_ankle', 'left_small_toe'), id=22, color=[0, 255, 0]),
        23:
        dict(link=('left_ankle', 'left_heel'), id=23, color=[0, 255, 0]),
        24:
        dict(
            link=('right_ankle', 'right_big_toe'), id=24, color=[255, 128, 0]),
        25:
        dict(
            link=('right_ankle', 'right_small_toe'),
            id=25,
            color=[255, 128, 0]),
        26:
        dict(link=('right_ankle', 'right_heel'), id=26, color=[255, 128, 0]),
    },
    # the joint_weights is modified by MMPose Team
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ] + [1., 1., 1.2] + [1.5] * 6,

    # 'https://github.com/Fang-Haoshu/Halpe-FullBody/blob/master/'
    # 'HalpeCOCOAPI/PythonAPI/halpecocotools/cocoeval.py#L245'
    sigmas=[
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
        0.026,
        0.026,
        0.066,
        0.079,
        0.079,
        0.079,
        0.079,
        0.079,
        0.079,
    ])
