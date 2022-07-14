dataset_info = dict(
    dataset_name='nyu',
    paper_info=dict(
        author='Jonathan Tompson and Murphy Stein and Yann Lecun and '
        'Ken Perlin',
        title='Real-Time Continuous Pose Recovery of Human Hands '
        'Using Convolutional Networks',
        container='ACM Transactions on Graphics',
        year='2014',
        homepage='https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm',
    ),
    keypoint_info={
        0: dict(name='F1_KNU3_A', id=0, color=[255, 128, 0], type='', swap=''),
        1: dict(name='F1_KNU3_B', id=1, color=[255, 128, 0], type='', swap=''),
        2: dict(name='F1_KNU2_A', id=2, color=[255, 128, 0], type='', swap=''),
        3: dict(name='F1_KNU2_B', id=3, color=[255, 128, 0], type='', swap=''),
        4:
        dict(name='F1_KNU1_A', id=4, color=[255, 153, 255], type='', swap=''),
        5:
        dict(name='F1_KNU1_B', id=5, color=[255, 153, 255], type='', swap=''),
        6:
        dict(name='F2_KNU3_A', id=6, color=[255, 153, 255], type='', swap=''),
        7:
        dict(name='F2_KNU3_B', id=7, color=[255, 153, 255], type='', swap=''),
        8:
        dict(name='F2_KNU2_A', id=8, color=[102, 178, 255], type='', swap=''),
        9:
        dict(name='F2_KNU2_B', id=9, color=[102, 178, 255], type='', swap=''),
        10:
        dict(name='F2_KNU1_A', id=10, color=[102, 178, 255], type='', swap=''),
        11:
        dict(name='F2_KNU1_B', id=11, color=[102, 178, 255], type='', swap=''),
        12:
        dict(name='F3_KNU3_A', id=12, color=[255, 51, 51], type='', swap=''),
        13:
        dict(name='F3_KNU3_B', id=13, color=[255, 51, 51], type='', swap=''),
        14:
        dict(name='F3_KNU2_A', id=14, color=[255, 51, 51], type='', swap=''),
        15:
        dict(name='F3_KNU2_B', id=15, color=[255, 51, 51], type='', swap=''),
        16: dict(name='F3_KNU1_A', id=16, color=[0, 255, 0], type='', swap=''),
        17: dict(name='F3_KNU1_B', id=17, color=[0, 255, 0], type='', swap=''),
        18: dict(name='F4_KNU3_A', id=18, color=[0, 255, 0], type='', swap=''),
        19: dict(name='F4_KNU3_B', id=19, color=[0, 255, 0], type='', swap=''),
        20:
        dict(name='F4_KNU2_A', id=20, color=[255, 255, 255], type='', swap=''),
        21:
        dict(name='F4_KNU2_B', id=21, color=[255, 128, 0], type='', swap=''),
        22:
        dict(name='F4_KNU1_A', id=22, color=[255, 128, 0], type='', swap=''),
        23:
        dict(name='F4_KNU1_B', id=23, color=[255, 128, 0], type='', swap=''),
        24:
        dict(name='TH_KNU3_A', id=24, color=[255, 128, 0], type='', swap=''),
        25:
        dict(name='TH_KNU3_B', id=25, color=[255, 153, 255], type='', swap=''),
        26:
        dict(name='TH_KNU2_A', id=26, color=[255, 153, 255], type='', swap=''),
        27:
        dict(name='TH_KNU2_B', id=27, color=[255, 153, 255], type='', swap=''),
        28:
        dict(name='TH_KNU1_A', id=28, color=[255, 153, 255], type='', swap=''),
        29:
        dict(name='TH_KNU1_B', id=29, color=[102, 178, 255], type='', swap=''),
        30:
        dict(name='PALM_1', id=30, color=[102, 178, 255], type='', swap=''),
        31:
        dict(name='PALM_2', id=31, color=[102, 178, 255], type='', swap=''),
        32:
        dict(name='PALM_3', id=32, color=[102, 178, 255], type='', swap=''),
        33: dict(name='PALM_4', id=33, color=[255, 51, 51], type='', swap=''),
        34: dict(name='PALM_5', id=34, color=[255, 51, 51], type='', swap=''),
        35: dict(name='PALM_6', id=35, color=[255, 51, 51], type='', swap=''),
    },
    skeleton_info={
        0: dict(link=('PALM_3', 'F1_KNU2_B'), id=0, color=[255, 128, 0]),
        1: dict(link=('F1_KNU2_B', 'F1_KNU3_A'), id=1, color=[255, 128, 0]),
        2: dict(link=('PALM_3', 'F2_KNU2_B'), id=2, color=[255, 128, 0]),
        3: dict(link=('F2_KNU2_B', 'F2_KNU3_A'), id=3, color=[255, 128, 0]),
        4: dict(link=('PALM_3', 'F3_KNU2_B'), id=4, color=[255, 153, 255]),
        5: dict(link=('F3_KNU2_B', 'F3_KNU3_A'), id=5, color=[255, 153, 255]),
        6: dict(link=('PALM_3', 'F4_KNU2_B'), id=6, color=[255, 153, 255]),
        7: dict(link=('F4_KNU2_B', 'F4_KNU3_A'), id=7, color=[255, 153, 255]),
        8: dict(link=('PALM_3', 'TH_KNU2_B'), id=8, color=[102, 178, 255]),
        9: dict(link=('TH_KNU2_B', 'TH_KNU3_B'), id=9, color=[102, 178, 255]),
        10:
        dict(link=('TH_KNU3_B', 'TH_KNU3_A'), id=10, color=[102, 178, 255]),
        11: dict(link=('PALM_3', 'PALM_1'), id=11, color=[102, 178, 255]),
        12: dict(link=('PALM_3', 'PALM_2'), id=12, color=[255, 51, 51]),
    },
    joint_weights=[1.] * 36,
    sigmas=[])
