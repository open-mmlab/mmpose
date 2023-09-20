dataset_info = dict(
    dataset_name='ubody3d',
    paper_info=dict(
        author='Jing Lin, Ailing Zeng, Haoqian Wang, Lei Zhang, Yu Li',
        title='One-Stage 3D Whole-Body Mesh Recovery with Component Aware'
        'Transformer',
        container='IEEE Computer Society Conference on Computer Vision and '
        'Pattern Recognition (CVPR)',
        year='2023',
        homepage='https://github.com/IDEA-Research/OSX',
    ),
    keypoint_info={
        0:
        dict(name='Pelvis', id=0, color=[0, 255, 0], type='', swap=''),
        1:
        dict(
            name='L_Hip', id=1, color=[0, 255, 0], type='lower', swap='R_Hip'),
        2:
        dict(
            name='R_Hip', id=2, color=[0, 255, 0], type='lower', swap='L_Hip'),
        3:
        dict(
            name='L_Knee',
            id=3,
            color=[0, 255, 0],
            type='lower',
            swap='R_Knee'),
        4:
        dict(
            name='R_Knee',
            id=4,
            color=[0, 255, 0],
            type='lower',
            swap='L_Knee'),
        5:
        dict(
            name='L_Ankle',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='R_Ankle'),
        6:
        dict(
            name='R_Ankle',
            id=6,
            color=[0, 255, 0],
            type='lower',
            swap='L_Ankle'),
        7:
        dict(name='Neck', id=7, color=[0, 255, 0], type='upper', swap=''),
        8:
        dict(
            name='L_Shoulder',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='R_Shoulder'),
        9:
        dict(
            name='R_Shoulder',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='L_Shoulder'),
        10:
        dict(
            name='L_Elbow',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='R_Elbow'),
        11:
        dict(
            name='R_Elbow',
            id=11,
            color=[0, 255, 0],
            type='upper',
            swap='L_Elbow'),
        12:
        dict(
            name='L_Wrist',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='R_Wrist'),
        13:
        dict(
            name='R_Wrist',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='L_Wrist'),
        14:
        dict(
            name='L_Big_toe',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='R_Big_toe'),
        15:
        dict(
            name='L_Small_toe',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='R_Small_toe'),
        16:
        dict(
            name='L_Heel',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='R_Heel'),
        17:
        dict(
            name='R_Big_toe',
            id=17,
            color=[0, 255, 0],
            type='lower',
            swap='L_Big_toe'),
        18:
        dict(
            name='R_Small_toe',
            id=18,
            color=[0, 255, 0],
            type='lower',
            swap='L_Small_toe'),
        19:
        dict(
            name='R_Heel',
            id=19,
            color=[0, 255, 0],
            type='lower',
            swap='L_Heel'),
        20:
        dict(
            name='L_Ear', id=20, color=[0, 255, 0], type='upper',
            swap='R_Ear'),
        21:
        dict(
            name='R_Ear', id=21, color=[0, 255, 0], type='upper',
            swap='L_Ear'),
        22:
        dict(name='L_Eye', id=22, color=[0, 255, 0], type='', swap='R_Eye'),
        23:
        dict(name='R_Eye', id=23, color=[0, 255, 0], type='', swap='L_Eye'),
        24:
        dict(name='Nose', id=24, color=[0, 255, 0], type='upper', swap=''),
        25:
        dict(
            name='L_Thumb_1',
            id=25,
            color=[255, 128, 0],
            type='',
            swap='R_Thumb_1'),
        26:
        dict(
            name='L_Thumb_2',
            id=26,
            color=[255, 128, 0],
            type='',
            swap='R_Thumb_2'),
        27:
        dict(
            name='L_Thumb_3',
            id=27,
            color=[255, 128, 0],
            type='',
            swap='R_Thumb_3'),
        28:
        dict(
            name='L_Thumb_4',
            id=28,
            color=[255, 128, 0],
            type='',
            swap='R_Thumb_4'),
        29:
        dict(
            name='L_Index_1',
            id=29,
            color=[255, 128, 0],
            type='',
            swap='R_Index_1'),
        30:
        dict(
            name='L_Index_2',
            id=30,
            color=[255, 128, 0],
            type='',
            swap='R_Index_2'),
        31:
        dict(
            name='L_Index_3',
            id=31,
            color=[255, 128, 0],
            type='',
            swap='R_Index_3'),
        32:
        dict(
            name='L_Index_4',
            id=32,
            color=[255, 128, 0],
            type='',
            swap='R_Index_4'),
        33:
        dict(
            name='L_Middle_1',
            id=33,
            color=[255, 128, 0],
            type='',
            swap='R_Middle_1'),
        34:
        dict(
            name='L_Middle_2',
            id=34,
            color=[255, 128, 0],
            type='',
            swap='R_Middle_2'),
        35:
        dict(
            name='L_Middle_3',
            id=35,
            color=[255, 128, 0],
            type='',
            swap='R_Middle_3'),
        36:
        dict(
            name='L_Middle_4',
            id=36,
            color=[255, 128, 0],
            type='',
            swap='R_Middle_4'),
        37:
        dict(
            name='L_Ring_1',
            id=37,
            color=[255, 128, 0],
            type='',
            swap='R_Ring_1'),
        38:
        dict(
            name='L_Ring_2',
            id=38,
            color=[255, 128, 0],
            type='',
            swap='R_Ring_2'),
        39:
        dict(
            name='L_Ring_3',
            id=39,
            color=[255, 128, 0],
            type='',
            swap='R_Ring_3'),
        40:
        dict(
            name='L_Ring_4',
            id=40,
            color=[255, 128, 0],
            type='',
            swap='R_Ring_4'),
        41:
        dict(
            name='L_Pinky_1',
            id=41,
            color=[255, 128, 0],
            type='',
            swap='R_Pinky_1'),
        42:
        dict(
            name='L_Pinky_2',
            id=42,
            color=[255, 128, 0],
            type='',
            swap='R_Pinky_2'),
        43:
        dict(
            name='L_Pinky_3',
            id=43,
            color=[255, 128, 0],
            type='',
            swap='R_Pinky_3'),
        44:
        dict(
            name='L_Pinky_4',
            id=44,
            color=[255, 128, 0],
            type='',
            swap='R_Pinky_4'),
        45:
        dict(
            name='R_Thumb_1',
            id=45,
            color=[255, 128, 0],
            type='',
            swap='L_Thumb_1'),
        46:
        dict(
            name='R_Thumb_2',
            id=46,
            color=[255, 128, 0],
            type='',
            swap='L_Thumb_2'),
        47:
        dict(
            name='R_Thumb_3',
            id=47,
            color=[255, 128, 0],
            type='',
            swap='L_Thumb_3'),
        48:
        dict(
            name='R_Thumb_4',
            id=48,
            color=[255, 128, 0],
            type='',
            swap='L_Thumb_4'),
        49:
        dict(
            name='R_Index_1',
            id=49,
            color=[255, 128, 0],
            type='',
            swap='L_Index_1'),
        50:
        dict(
            name='R_Index_2',
            id=50,
            color=[255, 128, 0],
            type='',
            swap='L_Index_2'),
        51:
        dict(
            name='R_Index_3',
            id=51,
            color=[255, 128, 0],
            type='',
            swap='L_Index_3'),
        52:
        dict(
            name='R_Index_4',
            id=52,
            color=[255, 128, 0],
            type='',
            swap='L_Index_4'),
        53:
        dict(
            name='R_Middle_1',
            id=53,
            color=[255, 128, 0],
            type='',
            swap='L_Middle_1'),
        54:
        dict(
            name='R_Middle_2',
            id=54,
            color=[255, 128, 0],
            type='',
            swap='L_Middle_2'),
        55:
        dict(
            name='R_Middle_3',
            id=55,
            color=[255, 128, 0],
            type='',
            swap='L_Middle_3'),
        56:
        dict(
            name='R_Middle_4',
            id=56,
            color=[255, 128, 0],
            type='',
            swap='L_Middle_4'),
        57:
        dict(
            name='R_Ring_1',
            id=57,
            color=[255, 128, 0],
            type='',
            swap='L_Ring_1'),
        58:
        dict(
            name='R_Ring_2',
            id=58,
            color=[255, 128, 0],
            type='',
            swap='L_Ring_2'),
        59:
        dict(
            name='R_Ring_3',
            id=59,
            color=[255, 128, 0],
            type='',
            swap='L_Ring_3'),
        60:
        dict(
            name='R_Ring_4',
            id=60,
            color=[255, 128, 0],
            type='',
            swap='L_Ring_4'),
        61:
        dict(
            name='R_Pinky_1',
            id=61,
            color=[255, 128, 0],
            type='',
            swap='L_Pinky_1'),
        62:
        dict(
            name='R_Pinky_2',
            id=62,
            color=[255, 128, 0],
            type='',
            swap='L_Pinky_2'),
        63:
        dict(
            name='R_Pinky_3',
            id=63,
            color=[255, 128, 0],
            type='',
            swap='L_Pinky_3'),
        64:
        dict(
            name='R_Pinky_4',
            id=64,
            color=[255, 128, 0],
            type='',
            swap='L_Pinky_4'),
        65:
        dict(name='Face_1', id=65, color=[255, 255, 255], type='', swap=''),
        66:
        dict(name='Face_2', id=66, color=[255, 255, 255], type='', swap=''),
        67:
        dict(
            name='Face_3',
            id=67,
            color=[255, 255, 255],
            type='',
            swap='Face_4'),
        68:
        dict(
            name='Face_4',
            id=68,
            color=[255, 255, 255],
            type='',
            swap='Face_3'),
        69:
        dict(
            name='Face_5',
            id=69,
            color=[255, 255, 255],
            type='',
            swap='Face_14'),
        70:
        dict(
            name='Face_6',
            id=70,
            color=[255, 255, 255],
            type='',
            swap='Face_13'),
        71:
        dict(
            name='Face_7',
            id=71,
            color=[255, 255, 255],
            type='',
            swap='Face_12'),
        72:
        dict(
            name='Face_8',
            id=72,
            color=[255, 255, 255],
            type='',
            swap='Face_11'),
        73:
        dict(
            name='Face_9',
            id=73,
            color=[255, 255, 255],
            type='',
            swap='Face_10'),
        74:
        dict(
            name='Face_10',
            id=74,
            color=[255, 255, 255],
            type='',
            swap='Face_9'),
        75:
        dict(
            name='Face_11',
            id=75,
            color=[255, 255, 255],
            type='',
            swap='Face_8'),
        76:
        dict(
            name='Face_12',
            id=76,
            color=[255, 255, 255],
            type='',
            swap='Face_7'),
        77:
        dict(
            name='Face_13',
            id=77,
            color=[255, 255, 255],
            type='',
            swap='Face_6'),
        78:
        dict(
            name='Face_14',
            id=78,
            color=[255, 255, 255],
            type='',
            swap='Face_5'),
        79:
        dict(name='Face_15', id=79, color=[255, 255, 255], type='', swap=''),
        80:
        dict(name='Face_16', id=80, color=[255, 255, 255], type='', swap=''),
        81:
        dict(name='Face_17', id=81, color=[255, 255, 255], type='', swap=''),
        82:
        dict(name='Face_18', id=82, color=[255, 255, 255], type='', swap=''),
        83:
        dict(
            name='Face_19',
            id=83,
            color=[255, 255, 255],
            type='',
            swap='Face_23'),
        84:
        dict(
            name='Face_20',
            id=84,
            color=[255, 255, 255],
            type='',
            swap='Face_22'),
        85:
        dict(name='Face_21', id=85, color=[255, 255, 255], type='', swap=''),
        86:
        dict(
            name='Face_22',
            id=86,
            color=[255, 255, 255],
            type='',
            swap='Face_20'),
        87:
        dict(
            name='Face_23',
            id=87,
            color=[255, 255, 255],
            type='',
            swap='Face_19'),
        88:
        dict(
            name='Face_24',
            id=88,
            color=[255, 255, 255],
            type='',
            swap='Face_33'),
        89:
        dict(
            name='Face_25',
            id=89,
            color=[255, 255, 255],
            type='',
            swap='Face_32'),
        90:
        dict(
            name='Face_26',
            id=90,
            color=[255, 255, 255],
            type='',
            swap='Face_31'),
        91:
        dict(
            name='Face_27',
            id=91,
            color=[255, 255, 255],
            type='',
            swap='Face_30'),
        92:
        dict(
            name='Face_28',
            id=92,
            color=[255, 255, 255],
            type='',
            swap='Face_35'),
        93:
        dict(
            name='Face_29',
            id=93,
            color=[255, 255, 255],
            type='',
            swap='Face_34'),
        94:
        dict(
            name='Face_30',
            id=94,
            color=[255, 255, 255],
            type='',
            swap='Face_27'),
        95:
        dict(
            name='Face_31',
            id=95,
            color=[255, 255, 255],
            type='',
            swap='Face_26'),
        96:
        dict(
            name='Face_32',
            id=96,
            color=[255, 255, 255],
            type='',
            swap='Face_25'),
        97:
        dict(
            name='Face_33',
            id=97,
            color=[255, 255, 255],
            type='',
            swap='Face_24'),
        98:
        dict(
            name='Face_34',
            id=98,
            color=[255, 255, 255],
            type='',
            swap='Face_29'),
        99:
        dict(
            name='Face_35',
            id=99,
            color=[255, 255, 255],
            type='',
            swap='Face_28'),
        100:
        dict(
            name='Face_36',
            id=100,
            color=[255, 255, 255],
            type='',
            swap='Face_42'),
        101:
        dict(
            name='Face_37',
            id=101,
            color=[255, 255, 255],
            type='',
            swap='Face_41'),
        102:
        dict(
            name='Face_38',
            id=102,
            color=[255, 255, 255],
            type='',
            swap='Face_40'),
        103:
        dict(name='Face_39', id=103, color=[255, 255, 255], type='', swap=''),
        104:
        dict(
            name='Face_40',
            id=104,
            color=[255, 255, 255],
            type='',
            swap='Face_38'),
        105:
        dict(
            name='Face_41',
            id=105,
            color=[255, 255, 255],
            type='',
            swap='Face_37'),
        106:
        dict(
            name='Face_42',
            id=106,
            color=[255, 255, 255],
            type='',
            swap='Face_36'),
        107:
        dict(
            name='Face_43',
            id=107,
            color=[255, 255, 255],
            type='',
            swap='Face_47'),
        108:
        dict(
            name='Face_44',
            id=108,
            color=[255, 255, 255],
            type='',
            swap='Face_46'),
        109:
        dict(name='Face_45', id=109, color=[255, 255, 255], type='', swap=''),
        110:
        dict(
            name='Face_46',
            id=110,
            color=[255, 255, 255],
            type='',
            swap='Face_44'),
        111:
        dict(
            name='Face_47',
            id=111,
            color=[255, 255, 255],
            type='',
            swap='Face_43'),
        112:
        dict(
            name='Face_48',
            id=112,
            color=[255, 255, 255],
            type='',
            swap='Face_52'),
        113:
        dict(
            name='Face_49',
            id=113,
            color=[255, 255, 255],
            type='',
            swap='Face_51'),
        114:
        dict(name='Face_50', id=114, color=[255, 255, 255], type='', swap=''),
        115:
        dict(
            name='Face_51',
            id=115,
            color=[255, 255, 255],
            type='',
            swap='Face_49'),
        116:
        dict(
            name='Face_52',
            id=116,
            color=[255, 255, 255],
            type='',
            swap='Face_48'),
        117:
        dict(
            name='Face_53',
            id=117,
            color=[255, 255, 255],
            type='',
            swap='Face_55'),
        118:
        dict(name='Face_54', id=118, color=[255, 255, 255], type='', swap=''),
        119:
        dict(
            name='Face_55',
            id=119,
            color=[255, 255, 255],
            type='',
            swap='Face_53'),
        120:
        dict(
            name='Face_56',
            id=120,
            color=[255, 255, 255],
            type='',
            swap='Face_72'),
        121:
        dict(
            name='Face_57',
            id=121,
            color=[255, 255, 255],
            type='',
            swap='Face_71'),
        122:
        dict(
            name='Face_58',
            id=122,
            color=[255, 255, 255],
            type='',
            swap='Face_70'),
        123:
        dict(
            name='Face_59',
            id=123,
            color=[255, 255, 255],
            type='',
            swap='Face_69'),
        124:
        dict(
            name='Face_60',
            id=124,
            color=[255, 255, 255],
            type='',
            swap='Face_68'),
        125:
        dict(
            name='Face_61',
            id=125,
            color=[255, 255, 255],
            type='',
            swap='Face_67'),
        126:
        dict(
            name='Face_62',
            id=126,
            color=[255, 255, 255],
            type='',
            swap='Face_66'),
        127:
        dict(
            name='Face_63',
            id=127,
            color=[255, 255, 255],
            type='',
            swap='Face_65'),
        128:
        dict(name='Face_64', id=128, color=[255, 255, 255], type='', swap=''),
        129:
        dict(
            name='Face_65',
            id=129,
            color=[255, 255, 255],
            type='',
            swap='Face_63'),
        130:
        dict(
            name='Face_66',
            id=130,
            color=[255, 255, 255],
            type='',
            swap='Face_62'),
        131:
        dict(
            name='Face_67',
            id=131,
            color=[255, 255, 255],
            type='',
            swap='Face_61'),
        132:
        dict(
            name='Face_68',
            id=132,
            color=[255, 255, 255],
            type='',
            swap='Face_60'),
        133:
        dict(
            name='Face_69',
            id=133,
            color=[255, 255, 255],
            type='',
            swap='Face_59'),
        134:
        dict(
            name='Face_70',
            id=134,
            color=[255, 255, 255],
            type='',
            swap='Face_58'),
        135:
        dict(
            name='Face_71',
            id=135,
            color=[255, 255, 255],
            type='',
            swap='Face_57'),
        136:
        dict(
            name='Face_72',
            id=136,
            color=[255, 255, 255],
            type='',
            swap='Face_56'),
    },
    skeleton_info={
        0: dict(link=('L_Ankle', 'L_Knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('L_Knee', 'L_Hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('R_Ankle', 'R_Knee'), id=2, color=[0, 255, 0]),
        3: dict(link=('R_Knee', 'R_Hip'), id=3, color=[0, 255, 0]),
        4: dict(link=('L_Hip', 'R_Hip'), id=4, color=[0, 255, 0]),
        5: dict(link=('L_Shoulder', 'L_Hip'), id=5, color=[0, 255, 0]),
        6: dict(link=('R_Shoulder', 'R_Hip'), id=6, color=[0, 255, 0]),
        7: dict(link=('L_Shoulder', 'R_Shoulder'), id=7, color=[0, 255, 0]),
        8: dict(link=('L_Shoulder', 'L_Elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('R_Shoulder', 'R_Elbow'), id=9, color=[0, 255, 0]),
        10: dict(link=('L_Elbow', 'L_Wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('R_Elbow', 'R_Wrist'), id=11, color=[255, 128, 0]),
        12: dict(link=('L_Eye', 'R_Eye'), id=12, color=[255, 128, 0]),
        13: dict(link=('Nose', 'L_Eye'), id=13, color=[255, 128, 0]),
        14: dict(link=('Nose', 'R_Eye'), id=14, color=[255, 128, 0]),
        15: dict(link=('L_Eye', 'L_Ear'), id=15, color=[255, 128, 0]),
        16: dict(link=('R_Eye', 'R_Ear'), id=16, color=[255, 128, 0]),
        17: dict(link=('L_Ear', 'L_Shoulder'), id=17, color=[255, 128, 0]),
        18: dict(link=('R_Ear', 'R_Shoulder'), id=18, color=[255, 128, 0]),
        19: dict(link=('L_Ankle', 'L_Big_toe'), id=19, color=[255, 128, 0]),
        20: dict(link=('L_Ankle', 'L_Small_toe'), id=20, color=[255, 128, 0]),
        21: dict(link=('L_Ankle', 'L_Heel'), id=21, color=[255, 128, 0]),
        22: dict(link=('R_Ankle', 'R_Big_toe'), id=22, color=[255, 128, 0]),
        23: dict(link=('R_Ankle', 'R_Small_toe'), id=23, color=[255, 128, 0]),
        24: dict(link=('R_Ankle', 'R_Heel'), id=24, color=[255, 128, 0]),
        25: dict(link=('L_Wrist', 'L_Thumb_1'), id=25, color=[255, 128, 0]),
        26: dict(link=('L_Thumb_1', 'L_Thumb_2'), id=26, color=[255, 128, 0]),
        27: dict(link=('L_Thumb_2', 'L_Thumb_3'), id=27, color=[255, 128, 0]),
        28: dict(link=('L_Thumb_3', 'L_Thumb_4'), id=28, color=[255, 128, 0]),
        29: dict(link=('L_Wrist', 'L_Index_1'), id=29, color=[255, 128, 0]),
        30: dict(link=('L_Index_1', 'L_Index_2'), id=30, color=[255, 128, 0]),
        31:
        dict(link=('L_Index_2', 'L_Index_3'), id=31, color=[255, 255, 255]),
        32:
        dict(link=('L_Index_3', 'L_Index_4'), id=32, color=[255, 255, 255]),
        33: dict(link=('L_Wrist', 'L_Middle_1'), id=33, color=[255, 255, 255]),
        34:
        dict(link=('L_Middle_1', 'L_Middle_2'), id=34, color=[255, 255, 255]),
        35:
        dict(link=('L_Middle_2', 'L_Middle_3'), id=35, color=[255, 255, 255]),
        36:
        dict(link=('L_Middle_3', 'L_Middle_4'), id=36, color=[255, 255, 255]),
        37: dict(link=('L_Wrist', 'L_Ring_1'), id=37, color=[255, 255, 255]),
        38: dict(link=('L_Ring_1', 'L_Ring_2'), id=38, color=[255, 255, 255]),
        39: dict(link=('L_Ring_2', 'L_Ring_3'), id=39, color=[255, 255, 255]),
        40: dict(link=('L_Ring_3', 'L_Ring_4'), id=40, color=[255, 255, 255]),
        41: dict(link=('L_Wrist', 'L_Pinky_1'), id=41, color=[255, 255, 255]),
        42:
        dict(link=('L_Pinky_1', 'L_Pinky_2'), id=42, color=[255, 255, 255]),
        43:
        dict(link=('L_Pinky_2', 'L_Pinky_3'), id=43, color=[255, 255, 255]),
        44:
        dict(link=('L_Pinky_3', 'L_Pinky_4'), id=44, color=[255, 255, 255]),
        45: dict(link=('R_Wrist', 'R_Thumb_1'), id=45, color=[255, 255, 255]),
        46:
        dict(link=('R_Thumb_1', 'R_Thumb_2'), id=46, color=[255, 255, 255]),
        47:
        dict(link=('R_Thumb_2', 'R_Thumb_3'), id=47, color=[255, 255, 255]),
        48:
        dict(link=('R_Thumb_3', 'R_Thumb_4'), id=48, color=[255, 255, 255]),
        49: dict(link=('R_Wrist', 'R_Index_1'), id=49, color=[255, 255, 255]),
        50:
        dict(link=('R_Index_1', 'R_Index_2'), id=50, color=[255, 255, 255]),
        51:
        dict(link=('R_Index_2', 'R_Index_3'), id=51, color=[255, 255, 255]),
        52:
        dict(link=('R_Index_3', 'R_Index_4'), id=52, color=[255, 255, 255]),
        53: dict(link=('R_Wrist', 'R_Middle_1'), id=53, color=[255, 255, 255]),
        54:
        dict(link=('R_Middle_1', 'R_Middle_2'), id=54, color=[255, 255, 255]),
        55:
        dict(link=('R_Middle_2', 'R_Middle_3'), id=55, color=[255, 255, 255]),
        56:
        dict(link=('R_Middle_3', 'R_Middle_4'), id=56, color=[255, 255, 255]),
        57: dict(link=('R_Wrist', 'R_Pinky_1'), id=57, color=[255, 255, 255]),
        58:
        dict(link=('R_Pinky_1', 'R_Pinky_2'), id=58, color=[255, 255, 255]),
        59:
        dict(link=('R_Pinky_2', 'R_Pinky_3'), id=59, color=[255, 255, 255]),
        60:
        dict(link=('R_Pinky_3', 'R_Pinky_4'), id=60, color=[255, 255, 255]),
    },
    joint_weights=[1.] * 137,
    sigmas=[])
