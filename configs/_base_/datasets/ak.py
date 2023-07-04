dataset_info = dict(
    dataset_name='Animal Kingdom',
    paper_info=dict(
        author='Singapore University of Technology and Design, Singapore.'
        ' Xun Long Ng, Kian Eng Ong, Qichen Zheng,'
        ' Yun Ni, Si Yong Yeo, Jun Liu.',
        title='Animal Kingdom: '
        'A Large and Diverse Dataset for Animal Behavior Understanding',
        container='Conference on Computer Vision '
        'and Pattern Recognition (CVPR)',
        year='2022',
        homepage='https://sutdcv.github.io/Animal-Kingdom',
        version='1.0 (2022-06)',
        date_created='2022-06',
    ),
    keypoint_info={
        0:
        dict(
            name='Head_Mid_Top',
            id=0,
            color=(225, 0, 255),
            type='upper',
            swap=''),
        1:
        dict(
            name='Eye_Left',
            id=1,
            color=[220, 20, 60],
            type='upper',
            swap='Eye_Right'),
        2:
        dict(
            name='Eye_Right',
            id=2,
            color=[0, 255, 255],
            type='upper',
            swap='Eye_Left'),
        3:
        dict(
            name='Mouth_Front_Top',
            id=3,
            color=(0, 255, 42),
            type='upper',
            swap=''),
        4:
        dict(
            name='Mouth_Back_Left',
            id=4,
            color=[221, 160, 221],
            type='upper',
            swap='Mouth_Back_Right'),
        5:
        dict(
            name='Mouth_Back_Right',
            id=5,
            color=[135, 206, 250],
            type='upper',
            swap='Mouth_Back_Left'),
        6:
        dict(
            name='Mouth_Front_Bottom',
            id=6,
            color=[50, 205, 50],
            type='upper',
            swap=''),
        7:
        dict(
            name='Shoulder_Left',
            id=7,
            color=[255, 182, 193],
            type='upper',
            swap='Shoulder_Right'),
        8:
        dict(
            name='Shoulder_Right',
            id=8,
            color=[0, 191, 255],
            type='upper',
            swap='Shoulder_Left'),
        9:
        dict(
            name='Elbow_Left',
            id=9,
            color=[255, 105, 180],
            type='upper',
            swap='Elbow_Right'),
        10:
        dict(
            name='Elbow_Right',
            id=10,
            color=[30, 144, 255],
            type='upper',
            swap='Elbow_Left'),
        11:
        dict(
            name='Wrist_Left',
            id=11,
            color=[255, 20, 147],
            type='upper',
            swap='Wrist_Right'),
        12:
        dict(
            name='Wrist_Right',
            id=12,
            color=[0, 0, 255],
            type='upper',
            swap='Wrist_Left'),
        13:
        dict(
            name='Torso_Mid_Back',
            id=13,
            color=(185, 3, 221),
            type='upper',
            swap=''),
        14:
        dict(
            name='Hip_Left',
            id=14,
            color=[255, 215, 0],
            type='lower',
            swap='Hip_Right'),
        15:
        dict(
            name='Hip_Right',
            id=15,
            color=[147, 112, 219],
            type='lower',
            swap='Hip_Left'),
        16:
        dict(
            name='Knee_Left',
            id=16,
            color=[255, 165, 0],
            type='lower',
            swap='Knee_Right'),
        17:
        dict(
            name='Knee_Right',
            id=17,
            color=[138, 43, 226],
            type='lower',
            swap='Knee_Left'),
        18:
        dict(
            name='Ankle_Left',
            id=18,
            color=[255, 140, 0],
            type='lower',
            swap='Ankle_Right'),
        19:
        dict(
            name='Ankle_Right',
            id=19,
            color=[128, 0, 128],
            type='lower',
            swap='Ankle_Left'),
        20:
        dict(
            name='Tail_Top_Back',
            id=20,
            color=(0, 251, 255),
            type='lower',
            swap=''),
        21:
        dict(
            name='Tail_Mid_Back',
            id=21,
            color=[32, 178, 170],
            type='lower',
            swap=''),
        22:
        dict(
            name='Tail_End_Back',
            id=22,
            color=(0, 102, 102),
            type='lower',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('Eye_Left', 'Head_Mid_Top'), id=0, color=[220, 20, 60]),
        1:
        dict(link=('Eye_Right', 'Head_Mid_Top'), id=1, color=[0, 255, 255]),
        2:
        dict(
            link=('Mouth_Front_Top', 'Mouth_Back_Left'),
            id=2,
            color=[221, 160, 221]),
        3:
        dict(
            link=('Mouth_Front_Top', 'Mouth_Back_Right'),
            id=3,
            color=[135, 206, 250]),
        4:
        dict(
            link=('Mouth_Front_Bottom', 'Mouth_Back_Left'),
            id=4,
            color=[221, 160, 221]),
        5:
        dict(
            link=('Mouth_Front_Bottom', 'Mouth_Back_Right'),
            id=5,
            color=[135, 206, 250]),
        6:
        dict(
            link=('Head_Mid_Top', 'Torso_Mid_Back'), id=6,
            color=(225, 0, 255)),
        7:
        dict(
            link=('Torso_Mid_Back', 'Tail_Top_Back'),
            id=7,
            color=(185, 3, 221)),
        8:
        dict(
            link=('Tail_Top_Back', 'Tail_Mid_Back'), id=8,
            color=(0, 251, 255)),
        9:
        dict(
            link=('Tail_Mid_Back', 'Tail_End_Back'),
            id=9,
            color=[32, 178, 170]),
        10:
        dict(
            link=('Head_Mid_Top', 'Shoulder_Left'),
            id=10,
            color=[255, 182, 193]),
        11:
        dict(
            link=('Head_Mid_Top', 'Shoulder_Right'),
            id=11,
            color=[0, 191, 255]),
        12:
        dict(
            link=('Shoulder_Left', 'Elbow_Left'), id=12, color=[255, 105,
                                                                180]),
        13:
        dict(
            link=('Shoulder_Right', 'Elbow_Right'),
            id=13,
            color=[30, 144, 255]),
        14:
        dict(link=('Elbow_Left', 'Wrist_Left'), id=14, color=[255, 20, 147]),
        15:
        dict(link=('Elbow_Right', 'Wrist_Right'), id=15, color=[0, 0, 255]),
        16:
        dict(link=('Tail_Top_Back', 'Hip_Left'), id=16, color=[255, 215, 0]),
        17:
        dict(
            link=('Tail_Top_Back', 'Hip_Right'), id=17, color=[147, 112, 219]),
        18:
        dict(link=('Hip_Left', 'Knee_Left'), id=18, color=[255, 165, 0]),
        19:
        dict(link=('Hip_Right', 'Knee_Right'), id=19, color=[138, 43, 226]),
        20:
        dict(link=('Knee_Left', 'Ankle_Left'), id=20, color=[255, 140, 0]),
        21:
        dict(link=('Knee_Right', 'Ankle_Right'), id=21, color=[128, 0, 128])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
        0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
        0.025, 0.025, 0.025
    ])
