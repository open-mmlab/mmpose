colors = dict(
    sss=[255, 128, 0],  # short_sleeve_shirt
    lss=[255, 0, 128],  # long_sleeved_shirt
    sso=[128, 0, 255],  # short_sleeved_outwear
    lso=[0, 128, 255],  # long_sleeved_outwear
    vest=[0, 128, 128],  # vest
    sling=[0, 0, 128],  # sling
    shorts=[128, 128, 128],  # shorts
    trousers=[128, 0, 128],  # trousers
    skirt=[64, 128, 128],  # skirt
    ssd=[64, 64, 128],  # short_sleeved_dress
    lsd=[128, 64, 0],  # long_sleeved_dress
    vd=[128, 64, 255],  # vest_dress
    sd=[128, 64, 0],  # sling_dress
)
dataset_info = dict(
    dataset_name='deepfashion2',
    paper_info=dict(
        author='Yuying Ge and Ruimao Zhang and Lingyun Wu '
        'and Xiaogang Wang and Xiaoou Tang and Ping Luo',
        title='DeepFashion2: A Versatile Benchmark for '
        'Detection, Pose Estimation, Segmentation and '
        'Re-Identification of Clothing Images',
        container='Proceedings of IEEE Conference on Computer '
        'Vision and Pattern Recognition (CVPR)',
        year='2019',
        homepage='https://github.com/switchablenorms/DeepFashion2',
    ),
    keypoint_info={
        # short_sleeved_shirt
        0:
        dict(name='sss_kpt1', id=0, color=colors['sss'], type='', swap=''),
        1:
        dict(
            name='sss_kpt2',
            id=1,
            color=colors['sss'],
            type='',
            swap='sss_kpt6'),
        2:
        dict(
            name='sss_kpt3',
            id=2,
            color=colors['sss'],
            type='',
            swap='sss_kpt5'),
        3:
        dict(name='sss_kpt4', id=3, color=colors['sss'], type='', swap=''),
        4:
        dict(
            name='sss_kpt5',
            id=4,
            color=colors['sss'],
            type='',
            swap='sss_kpt3'),
        5:
        dict(
            name='sss_kpt6',
            id=5,
            color=colors['sss'],
            type='',
            swap='sss_kpt2'),
        6:
        dict(
            name='sss_kpt7',
            id=6,
            color=colors['sss'],
            type='',
            swap='sss_kpt25'),
        7:
        dict(
            name='sss_kpt8',
            id=7,
            color=colors['sss'],
            type='',
            swap='sss_kpt24'),
        8:
        dict(
            name='sss_kpt9',
            id=8,
            color=colors['sss'],
            type='',
            swap='sss_kpt23'),
        9:
        dict(
            name='sss_kpt10',
            id=9,
            color=colors['sss'],
            type='',
            swap='sss_kpt22'),
        10:
        dict(
            name='sss_kpt11',
            id=10,
            color=colors['sss'],
            type='',
            swap='sss_kpt21'),
        11:
        dict(
            name='sss_kpt12',
            id=11,
            color=colors['sss'],
            type='',
            swap='sss_kpt20'),
        12:
        dict(
            name='sss_kpt13',
            id=12,
            color=colors['sss'],
            type='',
            swap='sss_kpt19'),
        13:
        dict(
            name='sss_kpt14',
            id=13,
            color=colors['sss'],
            type='',
            swap='sss_kpt18'),
        14:
        dict(
            name='sss_kpt15',
            id=14,
            color=colors['sss'],
            type='',
            swap='sss_kpt17'),
        15:
        dict(name='sss_kpt16', id=15, color=colors['sss'], type='', swap=''),
        16:
        dict(
            name='sss_kpt17',
            id=16,
            color=colors['sss'],
            type='',
            swap='sss_kpt15'),
        17:
        dict(
            name='sss_kpt18',
            id=17,
            color=colors['sss'],
            type='',
            swap='sss_kpt14'),
        18:
        dict(
            name='sss_kpt19',
            id=18,
            color=colors['sss'],
            type='',
            swap='sss_kpt13'),
        19:
        dict(
            name='sss_kpt20',
            id=19,
            color=colors['sss'],
            type='',
            swap='sss_kpt12'),
        20:
        dict(
            name='sss_kpt21',
            id=20,
            color=colors['sss'],
            type='',
            swap='sss_kpt11'),
        21:
        dict(
            name='sss_kpt22',
            id=21,
            color=colors['sss'],
            type='',
            swap='sss_kpt10'),
        22:
        dict(
            name='sss_kpt23',
            id=22,
            color=colors['sss'],
            type='',
            swap='sss_kpt9'),
        23:
        dict(
            name='sss_kpt24',
            id=23,
            color=colors['sss'],
            type='',
            swap='sss_kpt8'),
        24:
        dict(
            name='sss_kpt25',
            id=24,
            color=colors['sss'],
            type='',
            swap='sss_kpt7'),
        # long_sleeved_shirt
        25:
        dict(name='lss_kpt1', id=25, color=colors['lss'], type='', swap=''),
        26:
        dict(
            name='lss_kpt2',
            id=26,
            color=colors['lss'],
            type='',
            swap='lss_kpt6'),
        27:
        dict(
            name='lss_kpt3',
            id=27,
            color=colors['lss'],
            type='',
            swap='lss_kpt5'),
        28:
        dict(name='lss_kpt4', id=28, color=colors['lss'], type='', swap=''),
        29:
        dict(
            name='lss_kpt5',
            id=29,
            color=colors['lss'],
            type='',
            swap='lss_kpt3'),
        30:
        dict(
            name='lss_kpt6',
            id=30,
            color=colors['lss'],
            type='',
            swap='lss_kpt2'),
        31:
        dict(
            name='lss_kpt7',
            id=31,
            color=colors['lss'],
            type='',
            swap='lss_kpt33'),
        32:
        dict(
            name='lss_kpt8',
            id=32,
            color=colors['lss'],
            type='',
            swap='lss_kpt32'),
        33:
        dict(
            name='lss_kpt9',
            id=33,
            color=colors['lss'],
            type='',
            swap='lss_kpt31'),
        34:
        dict(
            name='lss_kpt10',
            id=34,
            color=colors['lss'],
            type='',
            swap='lss_kpt30'),
        35:
        dict(
            name='lss_kpt11',
            id=35,
            color=colors['lss'],
            type='',
            swap='lss_kpt29'),
        36:
        dict(
            name='lss_kpt12',
            id=36,
            color=colors['lss'],
            type='',
            swap='lss_kpt28'),
        37:
        dict(
            name='lss_kpt13',
            id=37,
            color=colors['lss'],
            type='',
            swap='lss_kpt27'),
        38:
        dict(
            name='lss_kpt14',
            id=38,
            color=colors['lss'],
            type='',
            swap='lss_kpt26'),
        39:
        dict(
            name='lss_kpt15',
            id=39,
            color=colors['lss'],
            type='',
            swap='lss_kpt25'),
        40:
        dict(
            name='lss_kpt16',
            id=40,
            color=colors['lss'],
            type='',
            swap='lss_kpt24'),
        41:
        dict(
            name='lss_kpt17',
            id=41,
            color=colors['lss'],
            type='',
            swap='lss_kpt23'),
        42:
        dict(
            name='lss_kpt18',
            id=42,
            color=colors['lss'],
            type='',
            swap='lss_kpt22'),
        43:
        dict(
            name='lss_kpt19',
            id=43,
            color=colors['lss'],
            type='',
            swap='lss_kpt21'),
        44:
        dict(name='lss_kpt20', id=44, color=colors['lss'], type='', swap=''),
        45:
        dict(
            name='lss_kpt21',
            id=45,
            color=colors['lss'],
            type='',
            swap='lss_kpt19'),
        46:
        dict(
            name='lss_kpt22',
            id=46,
            color=colors['lss'],
            type='',
            swap='lss_kpt18'),
        47:
        dict(
            name='lss_kpt23',
            id=47,
            color=colors['lss'],
            type='',
            swap='lss_kpt17'),
        48:
        dict(
            name='lss_kpt24',
            id=48,
            color=colors['lss'],
            type='',
            swap='lss_kpt16'),
        49:
        dict(
            name='lss_kpt25',
            id=49,
            color=colors['lss'],
            type='',
            swap='lss_kpt15'),
        50:
        dict(
            name='lss_kpt26',
            id=50,
            color=colors['lss'],
            type='',
            swap='lss_kpt14'),
        51:
        dict(
            name='lss_kpt27',
            id=51,
            color=colors['lss'],
            type='',
            swap='lss_kpt13'),
        52:
        dict(
            name='lss_kpt28',
            id=52,
            color=colors['lss'],
            type='',
            swap='lss_kpt12'),
        53:
        dict(
            name='lss_kpt29',
            id=53,
            color=colors['lss'],
            type='',
            swap='lss_kpt11'),
        54:
        dict(
            name='lss_kpt30',
            id=54,
            color=colors['lss'],
            type='',
            swap='lss_kpt10'),
        55:
        dict(
            name='lss_kpt31',
            id=55,
            color=colors['lss'],
            type='',
            swap='lss_kpt9'),
        56:
        dict(
            name='lss_kpt32',
            id=56,
            color=colors['lss'],
            type='',
            swap='lss_kpt8'),
        57:
        dict(
            name='lss_kpt33',
            id=57,
            color=colors['lss'],
            type='',
            swap='lss_kpt7'),
        # short_sleeved_outwear
        58:
        dict(name='sso_kpt1', id=58, color=colors['sso'], type='', swap=''),
        59:
        dict(
            name='sso_kpt2',
            id=59,
            color=colors['sso'],
            type='',
            swap='sso_kpt26'),
        60:
        dict(
            name='sso_kpt3',
            id=60,
            color=colors['sso'],
            type='',
            swap='sso_kpt5'),
        61:
        dict(
            name='sso_kpt4',
            id=61,
            color=colors['sso'],
            type='',
            swap='sso_kpt6'),
        62:
        dict(
            name='sso_kpt5',
            id=62,
            color=colors['sso'],
            type='',
            swap='sso_kpt3'),
        63:
        dict(
            name='sso_kpt6',
            id=63,
            color=colors['sso'],
            type='',
            swap='sso_kpt4'),
        64:
        dict(
            name='sso_kpt7',
            id=64,
            color=colors['sso'],
            type='',
            swap='sso_kpt25'),
        65:
        dict(
            name='sso_kpt8',
            id=65,
            color=colors['sso'],
            type='',
            swap='sso_kpt24'),
        66:
        dict(
            name='sso_kpt9',
            id=66,
            color=colors['sso'],
            type='',
            swap='sso_kpt23'),
        67:
        dict(
            name='sso_kpt10',
            id=67,
            color=colors['sso'],
            type='',
            swap='sso_kpt22'),
        68:
        dict(
            name='sso_kpt11',
            id=68,
            color=colors['sso'],
            type='',
            swap='sso_kpt21'),
        69:
        dict(
            name='sso_kpt12',
            id=69,
            color=colors['sso'],
            type='',
            swap='sso_kpt20'),
        70:
        dict(
            name='sso_kpt13',
            id=70,
            color=colors['sso'],
            type='',
            swap='sso_kpt19'),
        71:
        dict(
            name='sso_kpt14',
            id=71,
            color=colors['sso'],
            type='',
            swap='sso_kpt18'),
        72:
        dict(
            name='sso_kpt15',
            id=72,
            color=colors['sso'],
            type='',
            swap='sso_kpt17'),
        73:
        dict(
            name='sso_kpt16',
            id=73,
            color=colors['sso'],
            type='',
            swap='sso_kpt29'),
        74:
        dict(
            name='sso_kpt17',
            id=74,
            color=colors['sso'],
            type='',
            swap='sso_kpt15'),
        75:
        dict(
            name='sso_kpt18',
            id=75,
            color=colors['sso'],
            type='',
            swap='sso_kpt14'),
        76:
        dict(
            name='sso_kpt19',
            id=76,
            color=colors['sso'],
            type='',
            swap='sso_kpt13'),
        77:
        dict(
            name='sso_kpt20',
            id=77,
            color=colors['sso'],
            type='',
            swap='sso_kpt12'),
        78:
        dict(
            name='sso_kpt21',
            id=78,
            color=colors['sso'],
            type='',
            swap='sso_kpt11'),
        79:
        dict(
            name='sso_kpt22',
            id=79,
            color=colors['sso'],
            type='',
            swap='sso_kpt10'),
        80:
        dict(
            name='sso_kpt23',
            id=80,
            color=colors['sso'],
            type='',
            swap='sso_kpt9'),
        81:
        dict(
            name='sso_kpt24',
            id=81,
            color=colors['sso'],
            type='',
            swap='sso_kpt8'),
        82:
        dict(
            name='sso_kpt25',
            id=82,
            color=colors['sso'],
            type='',
            swap='sso_kpt7'),
        83:
        dict(
            name='sso_kpt26',
            id=83,
            color=colors['sso'],
            type='',
            swap='sso_kpt2'),
        84:
        dict(
            name='sso_kpt27',
            id=84,
            color=colors['sso'],
            type='',
            swap='sso_kpt30'),
        85:
        dict(
            name='sso_kpt28',
            id=85,
            color=colors['sso'],
            type='',
            swap='sso_kpt31'),
        86:
        dict(
            name='sso_kpt29',
            id=86,
            color=colors['sso'],
            type='',
            swap='sso_kpt16'),
        87:
        dict(
            name='sso_kpt30',
            id=87,
            color=colors['sso'],
            type='',
            swap='sso_kpt27'),
        88:
        dict(
            name='sso_kpt31',
            id=88,
            color=colors['sso'],
            type='',
            swap='sso_kpt28'),
        # long_sleeved_outwear
        89:
        dict(name='lso_kpt1', id=89, color=colors['lso'], type='', swap=''),
        90:
        dict(
            name='lso_kpt2',
            id=90,
            color=colors['lso'],
            type='',
            swap='lso_kpt6'),
        91:
        dict(
            name='lso_kpt3',
            id=91,
            color=colors['lso'],
            type='',
            swap='lso_kpt5'),
        92:
        dict(
            name='lso_kpt4',
            id=92,
            color=colors['lso'],
            type='',
            swap='lso_kpt34'),
        93:
        dict(
            name='lso_kpt5',
            id=93,
            color=colors['lso'],
            type='',
            swap='lso_kpt3'),
        94:
        dict(
            name='lso_kpt6',
            id=94,
            color=colors['lso'],
            type='',
            swap='lso_kpt2'),
        95:
        dict(
            name='lso_kpt7',
            id=95,
            color=colors['lso'],
            type='',
            swap='lso_kpt33'),
        96:
        dict(
            name='lso_kpt8',
            id=96,
            color=colors['lso'],
            type='',
            swap='lso_kpt32'),
        97:
        dict(
            name='lso_kpt9',
            id=97,
            color=colors['lso'],
            type='',
            swap='lso_kpt31'),
        98:
        dict(
            name='lso_kpt10',
            id=98,
            color=colors['lso'],
            type='',
            swap='lso_kpt30'),
        99:
        dict(
            name='lso_kpt11',
            id=99,
            color=colors['lso'],
            type='',
            swap='lso_kpt29'),
        100:
        dict(
            name='lso_kpt12',
            id=100,
            color=colors['lso'],
            type='',
            swap='lso_kpt28'),
        101:
        dict(
            name='lso_kpt13',
            id=101,
            color=colors['lso'],
            type='',
            swap='lso_kpt27'),
        102:
        dict(
            name='lso_kpt14',
            id=102,
            color=colors['lso'],
            type='',
            swap='lso_kpt26'),
        103:
        dict(
            name='lso_kpt15',
            id=103,
            color=colors['lso'],
            type='',
            swap='lso_kpt25'),
        104:
        dict(
            name='lso_kpt16',
            id=104,
            color=colors['lso'],
            type='',
            swap='lso_kpt24'),
        105:
        dict(
            name='lso_kpt17',
            id=105,
            color=colors['lso'],
            type='',
            swap='lso_kpt23'),
        106:
        dict(
            name='lso_kpt18',
            id=106,
            color=colors['lso'],
            type='',
            swap='lso_kpt22'),
        107:
        dict(
            name='lso_kpt19',
            id=107,
            color=colors['lso'],
            type='',
            swap='lso_kpt21'),
        108:
        dict(
            name='lso_kpt20',
            id=108,
            color=colors['lso'],
            type='',
            swap='lso_kpt37'),
        109:
        dict(
            name='lso_kpt21',
            id=109,
            color=colors['lso'],
            type='',
            swap='lso_kpt19'),
        110:
        dict(
            name='lso_kpt22',
            id=110,
            color=colors['lso'],
            type='',
            swap='lso_kpt18'),
        111:
        dict(
            name='lso_kpt23',
            id=111,
            color=colors['lso'],
            type='',
            swap='lso_kpt17'),
        112:
        dict(
            name='lso_kpt24',
            id=112,
            color=colors['lso'],
            type='',
            swap='lso_kpt16'),
        113:
        dict(
            name='lso_kpt25',
            id=113,
            color=colors['lso'],
            type='',
            swap='lso_kpt15'),
        114:
        dict(
            name='lso_kpt26',
            id=114,
            color=colors['lso'],
            type='',
            swap='lso_kpt14'),
        115:
        dict(
            name='lso_kpt27',
            id=115,
            color=colors['lso'],
            type='',
            swap='lso_kpt13'),
        116:
        dict(
            name='lso_kpt28',
            id=116,
            color=colors['lso'],
            type='',
            swap='lso_kpt12'),
        117:
        dict(
            name='lso_kpt29',
            id=117,
            color=colors['lso'],
            type='',
            swap='lso_kpt11'),
        118:
        dict(
            name='lso_kpt30',
            id=118,
            color=colors['lso'],
            type='',
            swap='lso_kpt10'),
        119:
        dict(
            name='lso_kpt31',
            id=119,
            color=colors['lso'],
            type='',
            swap='lso_kpt9'),
        120:
        dict(
            name='lso_kpt32',
            id=120,
            color=colors['lso'],
            type='',
            swap='lso_kpt8'),
        121:
        dict(
            name='lso_kpt33',
            id=121,
            color=colors['lso'],
            type='',
            swap='lso_kpt7'),
        122:
        dict(
            name='lso_kpt34',
            id=122,
            color=colors['lso'],
            type='',
            swap='lso_kpt4'),
        123:
        dict(
            name='lso_kpt35',
            id=123,
            color=colors['lso'],
            type='',
            swap='lso_kpt38'),
        124:
        dict(
            name='lso_kpt36',
            id=124,
            color=colors['lso'],
            type='',
            swap='lso_kpt39'),
        125:
        dict(
            name='lso_kpt37',
            id=125,
            color=colors['lso'],
            type='',
            swap='lso_kpt20'),
        126:
        dict(
            name='lso_kpt38',
            id=126,
            color=colors['lso'],
            type='',
            swap='lso_kpt35'),
        127:
        dict(
            name='lso_kpt39',
            id=127,
            color=colors['lso'],
            type='',
            swap='lso_kpt36'),
        # vest
        128:
        dict(name='vest_kpt1', id=128, color=colors['vest'], type='', swap=''),
        129:
        dict(
            name='vest_kpt2',
            id=129,
            color=colors['vest'],
            type='',
            swap='vest_kpt6'),
        130:
        dict(
            name='vest_kpt3',
            id=130,
            color=colors['vest'],
            type='',
            swap='vest_kpt5'),
        131:
        dict(name='vest_kpt4', id=131, color=colors['vest'], type='', swap=''),
        132:
        dict(
            name='vest_kpt5',
            id=132,
            color=colors['vest'],
            type='',
            swap='vest_kpt3'),
        133:
        dict(
            name='vest_kpt6',
            id=133,
            color=colors['vest'],
            type='',
            swap='vest_kpt2'),
        134:
        dict(
            name='vest_kpt7',
            id=134,
            color=colors['vest'],
            type='',
            swap='vest_kpt15'),
        135:
        dict(
            name='vest_kpt8',
            id=135,
            color=colors['vest'],
            type='',
            swap='vest_kpt14'),
        136:
        dict(
            name='vest_kpt9',
            id=136,
            color=colors['vest'],
            type='',
            swap='vest_kpt13'),
        137:
        dict(
            name='vest_kpt10',
            id=137,
            color=colors['vest'],
            type='',
            swap='vest_kpt12'),
        138:
        dict(
            name='vest_kpt11', id=138, color=colors['vest'], type='', swap=''),
        139:
        dict(
            name='vest_kpt12',
            id=139,
            color=colors['vest'],
            type='',
            swap='vest_kpt10'),
        140:
        dict(
            name='vest_kpt13', id=140, color=colors['vest'], type='', swap=''),
        141:
        dict(
            name='vest_kpt14',
            id=141,
            color=colors['vest'],
            type='',
            swap='vest_kpt8'),
        142:
        dict(
            name='vest_kpt15',
            id=142,
            color=colors['vest'],
            type='',
            swap='vest_kpt7'),
        # sling
        143:
        dict(
            name='sling_kpt1', id=143, color=colors['sling'], type='',
            swap=''),
        144:
        dict(
            name='sling_kpt2',
            id=144,
            color=colors['sling'],
            type='',
            swap='sling_kpt6'),
        145:
        dict(
            name='sling_kpt3',
            id=145,
            color=colors['sling'],
            type='',
            swap='sling_kpt5'),
        146:
        dict(
            name='sling_kpt4', id=146, color=colors['sling'], type='',
            swap=''),
        147:
        dict(
            name='sling_kpt5',
            id=147,
            color=colors['sling'],
            type='',
            swap='sling_kpt3'),
        148:
        dict(
            name='sling_kpt6',
            id=148,
            color=colors['sling'],
            type='',
            swap='sling_kpt2'),
        149:
        dict(
            name='sling_kpt7',
            id=149,
            color=colors['sling'],
            type='',
            swap='sling_kpt15'),
        150:
        dict(
            name='sling_kpt8',
            id=150,
            color=colors['sling'],
            type='',
            swap='sling_kpt14'),
        151:
        dict(
            name='sling_kpt9',
            id=151,
            color=colors['sling'],
            type='',
            swap='sling_kpt13'),
        152:
        dict(
            name='sling_kpt10',
            id=152,
            color=colors['sling'],
            type='',
            swap='sling_kpt12'),
        153:
        dict(
            name='sling_kpt11',
            id=153,
            color=colors['sling'],
            type='',
            swap=''),
        154:
        dict(
            name='sling_kpt12',
            id=154,
            color=colors['sling'],
            type='',
            swap='sling_kpt10'),
        155:
        dict(
            name='sling_kpt13',
            id=155,
            color=colors['sling'],
            type='',
            swap='sling_kpt9'),
        156:
        dict(
            name='sling_kpt14',
            id=156,
            color=colors['sling'],
            type='',
            swap='sling_kpt8'),
        157:
        dict(
            name='sling_kpt15',
            id=157,
            color=colors['sling'],
            type='',
            swap='sling_kpt7'),
        # shorts
        158:
        dict(
            name='shorts_kpt1',
            id=158,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt3'),
        159:
        dict(
            name='shorts_kpt2',
            id=159,
            color=colors['shorts'],
            type='',
            swap=''),
        160:
        dict(
            name='shorts_kpt3',
            id=160,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt1'),
        161:
        dict(
            name='shorts_kpt4',
            id=161,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt10'),
        162:
        dict(
            name='shorts_kpt5',
            id=162,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt9'),
        163:
        dict(
            name='shorts_kpt6',
            id=163,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt8'),
        164:
        dict(
            name='shorts_kpt7',
            id=164,
            color=colors['shorts'],
            type='',
            swap=''),
        165:
        dict(
            name='shorts_kpt8',
            id=165,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt6'),
        166:
        dict(
            name='shorts_kpt9',
            id=166,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt5'),
        167:
        dict(
            name='shorts_kpt10',
            id=167,
            color=colors['shorts'],
            type='',
            swap='shorts_kpt4'),
        # trousers
        168:
        dict(
            name='trousers_kpt1',
            id=168,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt3'),
        169:
        dict(
            name='trousers_kpt2',
            id=169,
            color=colors['trousers'],
            type='',
            swap=''),
        170:
        dict(
            name='trousers_kpt3',
            id=170,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt1'),
        171:
        dict(
            name='trousers_kpt4',
            id=171,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt14'),
        172:
        dict(
            name='trousers_kpt5',
            id=172,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt13'),
        173:
        dict(
            name='trousers_kpt6',
            id=173,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt12'),
        174:
        dict(
            name='trousers_kpt7',
            id=174,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt11'),
        175:
        dict(
            name='trousers_kpt8',
            id=175,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt10'),
        176:
        dict(
            name='trousers_kpt9',
            id=176,
            color=colors['trousers'],
            type='',
            swap=''),
        177:
        dict(
            name='trousers_kpt10',
            id=177,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt8'),
        178:
        dict(
            name='trousers_kpt11',
            id=178,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt7'),
        179:
        dict(
            name='trousers_kpt12',
            id=179,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt6'),
        180:
        dict(
            name='trousers_kpt13',
            id=180,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt5'),
        181:
        dict(
            name='trousers_kpt14',
            id=181,
            color=colors['trousers'],
            type='',
            swap='trousers_kpt4'),
        # skirt
        182:
        dict(
            name='skirt_kpt1',
            id=182,
            color=colors['skirt'],
            type='',
            swap='skirt_kpt3'),
        183:
        dict(
            name='skirt_kpt2', id=183, color=colors['skirt'], type='',
            swap=''),
        184:
        dict(
            name='skirt_kpt3',
            id=184,
            color=colors['skirt'],
            type='',
            swap='skirt_kpt1'),
        185:
        dict(
            name='skirt_kpt4',
            id=185,
            color=colors['skirt'],
            type='',
            swap='skirt_kpt8'),
        186:
        dict(
            name='skirt_kpt5',
            id=186,
            color=colors['skirt'],
            type='',
            swap='skirt_kpt7'),
        187:
        dict(
            name='skirt_kpt6', id=187, color=colors['skirt'], type='',
            swap=''),
        188:
        dict(
            name='skirt_kpt7',
            id=188,
            color=colors['skirt'],
            type='',
            swap='skirt_kpt5'),
        189:
        dict(
            name='skirt_kpt8',
            id=189,
            color=colors['skirt'],
            type='',
            swap='skirt_kpt4'),
        # short_sleeved_dress
        190:
        dict(name='ssd_kpt1', id=190, color=colors['ssd'], type='', swap=''),
        191:
        dict(
            name='ssd_kpt2',
            id=191,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt6'),
        192:
        dict(
            name='ssd_kpt3',
            id=192,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt5'),
        193:
        dict(name='ssd_kpt4', id=193, color=colors['ssd'], type='', swap=''),
        194:
        dict(
            name='ssd_kpt5',
            id=194,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt3'),
        195:
        dict(
            name='ssd_kpt6',
            id=195,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt2'),
        196:
        dict(
            name='ssd_kpt7',
            id=196,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt29'),
        197:
        dict(
            name='ssd_kpt8',
            id=197,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt28'),
        198:
        dict(
            name='ssd_kpt9',
            id=198,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt27'),
        199:
        dict(
            name='ssd_kpt10',
            id=199,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt26'),
        200:
        dict(
            name='ssd_kpt11',
            id=200,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt25'),
        201:
        dict(
            name='ssd_kpt12',
            id=201,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt24'),
        202:
        dict(
            name='ssd_kpt13',
            id=202,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt23'),
        203:
        dict(
            name='ssd_kpt14',
            id=203,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt22'),
        204:
        dict(
            name='ssd_kpt15',
            id=204,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt21'),
        205:
        dict(
            name='ssd_kpt16',
            id=205,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt20'),
        206:
        dict(
            name='ssd_kpt17',
            id=206,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt19'),
        207:
        dict(name='ssd_kpt18', id=207, color=colors['ssd'], type='', swap=''),
        208:
        dict(
            name='ssd_kpt19',
            id=208,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt17'),
        209:
        dict(
            name='ssd_kpt20',
            id=209,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt16'),
        210:
        dict(
            name='ssd_kpt21',
            id=210,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt15'),
        211:
        dict(
            name='ssd_kpt22',
            id=211,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt14'),
        212:
        dict(
            name='ssd_kpt23',
            id=212,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt13'),
        213:
        dict(
            name='ssd_kpt24',
            id=213,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt12'),
        214:
        dict(
            name='ssd_kpt25',
            id=214,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt11'),
        215:
        dict(
            name='ssd_kpt26',
            id=215,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt10'),
        216:
        dict(
            name='ssd_kpt27',
            id=216,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt9'),
        217:
        dict(
            name='ssd_kpt28',
            id=217,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt8'),
        218:
        dict(
            name='ssd_kpt29',
            id=218,
            color=colors['ssd'],
            type='',
            swap='ssd_kpt7'),
        # long_sleeved_dress
        219:
        dict(name='lsd_kpt1', id=219, color=colors['lsd'], type='', swap=''),
        220:
        dict(
            name='lsd_kpt2',
            id=220,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt6'),
        221:
        dict(
            name='lsd_kpt3',
            id=221,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt5'),
        222:
        dict(name='lsd_kpt4', id=222, color=colors['lsd'], type='', swap=''),
        223:
        dict(
            name='lsd_kpt5',
            id=223,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt3'),
        224:
        dict(
            name='lsd_kpt6',
            id=224,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt2'),
        225:
        dict(
            name='lsd_kpt7',
            id=225,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt37'),
        226:
        dict(
            name='lsd_kpt8',
            id=226,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt36'),
        227:
        dict(
            name='lsd_kpt9',
            id=227,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt35'),
        228:
        dict(
            name='lsd_kpt10',
            id=228,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt34'),
        229:
        dict(
            name='lsd_kpt11',
            id=229,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt33'),
        230:
        dict(
            name='lsd_kpt12',
            id=230,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt32'),
        231:
        dict(
            name='lsd_kpt13',
            id=231,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt31'),
        232:
        dict(
            name='lsd_kpt14',
            id=232,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt30'),
        233:
        dict(
            name='lsd_kpt15',
            id=233,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt29'),
        234:
        dict(
            name='lsd_kpt16',
            id=234,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt28'),
        235:
        dict(
            name='lsd_kpt17',
            id=235,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt27'),
        236:
        dict(
            name='lsd_kpt18',
            id=236,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt26'),
        237:
        dict(
            name='lsd_kpt19',
            id=237,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt25'),
        238:
        dict(
            name='lsd_kpt20',
            id=238,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt24'),
        239:
        dict(
            name='lsd_kpt21',
            id=239,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt23'),
        240:
        dict(name='lsd_kpt22', id=240, color=colors['lsd'], type='', swap=''),
        241:
        dict(
            name='lsd_kpt23',
            id=241,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt21'),
        242:
        dict(
            name='lsd_kpt24',
            id=242,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt20'),
        243:
        dict(
            name='lsd_kpt25',
            id=243,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt19'),
        244:
        dict(
            name='lsd_kpt26',
            id=244,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt18'),
        245:
        dict(
            name='lsd_kpt27',
            id=245,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt17'),
        246:
        dict(
            name='lsd_kpt28',
            id=246,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt16'),
        247:
        dict(
            name='lsd_kpt29',
            id=247,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt15'),
        248:
        dict(
            name='lsd_kpt30',
            id=248,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt14'),
        249:
        dict(
            name='lsd_kpt31',
            id=249,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt13'),
        250:
        dict(
            name='lsd_kpt32',
            id=250,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt12'),
        251:
        dict(
            name='lsd_kpt33',
            id=251,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt11'),
        252:
        dict(
            name='lsd_kpt34',
            id=252,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt10'),
        253:
        dict(
            name='lsd_kpt35',
            id=253,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt9'),
        254:
        dict(
            name='lsd_kpt36',
            id=254,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt8'),
        255:
        dict(
            name='lsd_kpt37',
            id=255,
            color=colors['lsd'],
            type='',
            swap='lsd_kpt7'),
        # vest_dress
        256:
        dict(name='vd_kpt1', id=256, color=colors['vd'], type='', swap=''),
        257:
        dict(
            name='vd_kpt2',
            id=257,
            color=colors['vd'],
            type='',
            swap='vd_kpt6'),
        258:
        dict(
            name='vd_kpt3',
            id=258,
            color=colors['vd'],
            type='',
            swap='vd_kpt5'),
        259:
        dict(name='vd_kpt4', id=259, color=colors['vd'], type='', swap=''),
        260:
        dict(
            name='vd_kpt5',
            id=260,
            color=colors['vd'],
            type='',
            swap='vd_kpt3'),
        261:
        dict(
            name='vd_kpt6',
            id=261,
            color=colors['vd'],
            type='',
            swap='vd_kpt2'),
        262:
        dict(
            name='vd_kpt7',
            id=262,
            color=colors['vd'],
            type='',
            swap='vd_kpt19'),
        263:
        dict(
            name='vd_kpt8',
            id=263,
            color=colors['vd'],
            type='',
            swap='vd_kpt18'),
        264:
        dict(
            name='vd_kpt9',
            id=264,
            color=colors['vd'],
            type='',
            swap='vd_kpt17'),
        265:
        dict(
            name='vd_kpt10',
            id=265,
            color=colors['vd'],
            type='',
            swap='vd_kpt16'),
        266:
        dict(
            name='vd_kpt11',
            id=266,
            color=colors['vd'],
            type='',
            swap='vd_kpt15'),
        267:
        dict(
            name='vd_kpt12',
            id=267,
            color=colors['vd'],
            type='',
            swap='vd_kpt14'),
        268:
        dict(name='vd_kpt13', id=268, color=colors['vd'], type='', swap=''),
        269:
        dict(
            name='vd_kpt14',
            id=269,
            color=colors['vd'],
            type='',
            swap='vd_kpt12'),
        270:
        dict(
            name='vd_kpt15',
            id=270,
            color=colors['vd'],
            type='',
            swap='vd_kpt11'),
        271:
        dict(
            name='vd_kpt16',
            id=271,
            color=colors['vd'],
            type='',
            swap='vd_kpt10'),
        272:
        dict(
            name='vd_kpt17',
            id=272,
            color=colors['vd'],
            type='',
            swap='vd_kpt9'),
        273:
        dict(
            name='vd_kpt18',
            id=273,
            color=colors['vd'],
            type='',
            swap='vd_kpt8'),
        274:
        dict(
            name='vd_kpt19',
            id=274,
            color=colors['vd'],
            type='',
            swap='vd_kpt7'),
        # sling_dress
        275:
        dict(name='sd_kpt1', id=275, color=colors['sd'], type='', swap=''),
        276:
        dict(
            name='sd_kpt2',
            id=276,
            color=colors['sd'],
            type='',
            swap='sd_kpt6'),
        277:
        dict(
            name='sd_kpt3',
            id=277,
            color=colors['sd'],
            type='',
            swap='sd_kpt5'),
        278:
        dict(name='sd_kpt4', id=278, color=colors['sd'], type='', swap=''),
        279:
        dict(
            name='sd_kpt5',
            id=279,
            color=colors['sd'],
            type='',
            swap='sd_kpt3'),
        280:
        dict(
            name='sd_kpt6',
            id=280,
            color=colors['sd'],
            type='',
            swap='sd_kpt2'),
        281:
        dict(
            name='sd_kpt7',
            id=281,
            color=colors['sd'],
            type='',
            swap='sd_kpt19'),
        282:
        dict(
            name='sd_kpt8',
            id=282,
            color=colors['sd'],
            type='',
            swap='sd_kpt18'),
        283:
        dict(
            name='sd_kpt9',
            id=283,
            color=colors['sd'],
            type='',
            swap='sd_kpt17'),
        284:
        dict(
            name='sd_kpt10',
            id=284,
            color=colors['sd'],
            type='',
            swap='sd_kpt16'),
        285:
        dict(
            name='sd_kpt11',
            id=285,
            color=colors['sd'],
            type='',
            swap='sd_kpt15'),
        286:
        dict(
            name='sd_kpt12',
            id=286,
            color=colors['sd'],
            type='',
            swap='sd_kpt14'),
        287:
        dict(name='sd_kpt13', id=287, color=colors['sd'], type='', swap=''),
        288:
        dict(
            name='sd_kpt14',
            id=288,
            color=colors['sd'],
            type='',
            swap='sd_kpt12'),
        289:
        dict(
            name='sd_kpt15',
            id=289,
            color=colors['sd'],
            type='',
            swap='sd_kpt11'),
        290:
        dict(
            name='sd_kpt16',
            id=290,
            color=colors['sd'],
            type='',
            swap='sd_kpt10'),
        291:
        dict(
            name='sd_kpt17',
            id=291,
            color=colors['sd'],
            type='',
            swap='sd_kpt9'),
        292:
        dict(
            name='sd_kpt18',
            id=292,
            color=colors['sd'],
            type='',
            swap='sd_kpt8'),
        293:
        dict(
            name='sd_kpt19',
            id=293,
            color=colors['sd'],
            type='',
            swap='sd_kpt7'),
    },
    skeleton_info={
        # short_sleeved_shirt
        0:
        dict(link=('sss_kpt1', 'sss_kpt2'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('sss_kpt2', 'sss_kpt7'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('sss_kpt7', 'sss_kpt8'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('sss_kpt8', 'sss_kpt9'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('sss_kpt9', 'sss_kpt10'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('sss_kpt10', 'sss_kpt11'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('sss_kpt11', 'sss_kpt12'), id=6, color=[255, 128, 0]),
        7:
        dict(link=('sss_kpt12', 'sss_kpt13'), id=7, color=[255, 128, 0]),
        8:
        dict(link=('sss_kpt13', 'sss_kpt14'), id=8, color=[255, 128, 0]),
        9:
        dict(link=('sss_kpt14', 'sss_kpt15'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('sss_kpt15', 'sss_kpt16'), id=10, color=[255, 128, 0]),
        11:
        dict(link=('sss_kpt16', 'sss_kpt17'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('sss_kpt17', 'sss_kpt18'), id=12, color=[255, 128, 0]),
        13:
        dict(link=('sss_kpt18', 'sss_kpt19'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('sss_kpt19', 'sss_kpt20'), id=14, color=[255, 128, 0]),
        15:
        dict(link=('sss_kpt20', 'sss_kpt21'), id=15, color=[255, 128, 0]),
        16:
        dict(link=('sss_kpt21', 'sss_kpt22'), id=16, color=[255, 128, 0]),
        17:
        dict(link=('sss_kpt22', 'sss_kpt23'), id=17, color=[255, 128, 0]),
        18:
        dict(link=('sss_kpt23', 'sss_kpt24'), id=18, color=[255, 128, 0]),
        19:
        dict(link=('sss_kpt24', 'sss_kpt25'), id=19, color=[255, 128, 0]),
        20:
        dict(link=('sss_kpt25', 'sss_kpt6'), id=20, color=[255, 128, 0]),
        21:
        dict(link=('sss_kpt6', 'sss_kpt1'), id=21, color=[255, 128, 0]),
        22:
        dict(link=('sss_kpt2', 'sss_kpt3'), id=22, color=[255, 128, 0]),
        23:
        dict(link=('sss_kpt3', 'sss_kpt4'), id=23, color=[255, 128, 0]),
        24:
        dict(link=('sss_kpt4', 'sss_kpt5'), id=24, color=[255, 128, 0]),
        25:
        dict(link=('sss_kpt5', 'sss_kpt6'), id=25, color=[255, 128, 0]),
        # long_sleeve_shirt
        26:
        dict(link=('lss_kpt1', 'lss_kpt2'), id=26, color=[255, 0, 128]),
        27:
        dict(link=('lss_kpt2', 'lss_kpt7'), id=27, color=[255, 0, 128]),
        28:
        dict(link=('lss_kpt7', 'lss_kpt8'), id=28, color=[255, 0, 128]),
        29:
        dict(link=('lss_kpt8', 'lss_kpt9'), id=29, color=[255, 0, 128]),
        30:
        dict(link=('lss_kpt9', 'lss_kpt10'), id=30, color=[255, 0, 128]),
        31:
        dict(link=('lss_kpt10', 'lss_kpt11'), id=31, color=[255, 0, 128]),
        32:
        dict(link=('lss_kpt11', 'lss_kpt12'), id=32, color=[255, 0, 128]),
        33:
        dict(link=('lss_kpt12', 'lss_kpt13'), id=33, color=[255, 0, 128]),
        34:
        dict(link=('lss_kpt13', 'lss_kpt14'), id=34, color=[255, 0, 128]),
        35:
        dict(link=('lss_kpt14', 'lss_kpt15'), id=35, color=[255, 0, 128]),
        36:
        dict(link=('lss_kpt15', 'lss_kpt16'), id=36, color=[255, 0, 128]),
        37:
        dict(link=('lss_kpt16', 'lss_kpt17'), id=37, color=[255, 0, 128]),
        38:
        dict(link=('lss_kpt17', 'lss_kpt18'), id=38, color=[255, 0, 128]),
        39:
        dict(link=('lss_kpt18', 'lss_kpt19'), id=39, color=[255, 0, 128]),
        40:
        dict(link=('lss_kpt19', 'lss_kpt20'), id=40, color=[255, 0, 128]),
        41:
        dict(link=('lss_kpt20', 'lss_kpt21'), id=41, color=[255, 0, 128]),
        42:
        dict(link=('lss_kpt21', 'lss_kpt22'), id=42, color=[255, 0, 128]),
        43:
        dict(link=('lss_kpt22', 'lss_kpt23'), id=43, color=[255, 0, 128]),
        44:
        dict(link=('lss_kpt23', 'lss_kpt24'), id=44, color=[255, 0, 128]),
        45:
        dict(link=('lss_kpt24', 'lss_kpt25'), id=45, color=[255, 0, 128]),
        46:
        dict(link=('lss_kpt25', 'lss_kpt26'), id=46, color=[255, 0, 128]),
        47:
        dict(link=('lss_kpt26', 'lss_kpt27'), id=47, color=[255, 0, 128]),
        48:
        dict(link=('lss_kpt27', 'lss_kpt28'), id=48, color=[255, 0, 128]),
        49:
        dict(link=('lss_kpt28', 'lss_kpt29'), id=49, color=[255, 0, 128]),
        50:
        dict(link=('lss_kpt29', 'lss_kpt30'), id=50, color=[255, 0, 128]),
        51:
        dict(link=('lss_kpt30', 'lss_kpt31'), id=51, color=[255, 0, 128]),
        52:
        dict(link=('lss_kpt31', 'lss_kpt32'), id=52, color=[255, 0, 128]),
        53:
        dict(link=('lss_kpt32', 'lss_kpt33'), id=53, color=[255, 0, 128]),
        54:
        dict(link=('lss_kpt33', 'lss_kpt6'), id=54, color=[255, 0, 128]),
        55:
        dict(link=('lss_kpt6', 'lss_kpt5'), id=55, color=[255, 0, 128]),
        56:
        dict(link=('lss_kpt5', 'lss_kpt4'), id=56, color=[255, 0, 128]),
        57:
        dict(link=('lss_kpt4', 'lss_kpt3'), id=57, color=[255, 0, 128]),
        58:
        dict(link=('lss_kpt3', 'lss_kpt2'), id=58, color=[255, 0, 128]),
        59:
        dict(link=('lss_kpt6', 'lss_kpt1'), id=59, color=[255, 0, 128]),
        # short_sleeved_outwear
        60:
        dict(link=('sso_kpt1', 'sso_kpt4'), id=60, color=[128, 0, 255]),
        61:
        dict(link=('sso_kpt4', 'sso_kpt7'), id=61, color=[128, 0, 255]),
        62:
        dict(link=('sso_kpt7', 'sso_kpt8'), id=62, color=[128, 0, 255]),
        63:
        dict(link=('sso_kpt8', 'sso_kpt9'), id=63, color=[128, 0, 255]),
        64:
        dict(link=('sso_kpt9', 'sso_kpt10'), id=64, color=[128, 0, 255]),
        65:
        dict(link=('sso_kpt10', 'sso_kpt11'), id=65, color=[128, 0, 255]),
        66:
        dict(link=('sso_kpt11', 'sso_kpt12'), id=66, color=[128, 0, 255]),
        67:
        dict(link=('sso_kpt12', 'sso_kpt13'), id=67, color=[128, 0, 255]),
        68:
        dict(link=('sso_kpt13', 'sso_kpt14'), id=68, color=[128, 0, 255]),
        69:
        dict(link=('sso_kpt14', 'sso_kpt15'), id=69, color=[128, 0, 255]),
        70:
        dict(link=('sso_kpt15', 'sso_kpt16'), id=70, color=[128, 0, 255]),
        71:
        dict(link=('sso_kpt16', 'sso_kpt31'), id=71, color=[128, 0, 255]),
        72:
        dict(link=('sso_kpt31', 'sso_kpt30'), id=72, color=[128, 0, 255]),
        73:
        dict(link=('sso_kpt30', 'sso_kpt2'), id=73, color=[128, 0, 255]),
        74:
        dict(link=('sso_kpt2', 'sso_kpt3'), id=74, color=[128, 0, 255]),
        75:
        dict(link=('sso_kpt3', 'sso_kpt4'), id=75, color=[128, 0, 255]),
        76:
        dict(link=('sso_kpt1', 'sso_kpt6'), id=76, color=[128, 0, 255]),
        77:
        dict(link=('sso_kpt6', 'sso_kpt25'), id=77, color=[128, 0, 255]),
        78:
        dict(link=('sso_kpt25', 'sso_kpt24'), id=78, color=[128, 0, 255]),
        79:
        dict(link=('sso_kpt24', 'sso_kpt23'), id=79, color=[128, 0, 255]),
        80:
        dict(link=('sso_kpt23', 'sso_kpt22'), id=80, color=[128, 0, 255]),
        81:
        dict(link=('sso_kpt22', 'sso_kpt21'), id=81, color=[128, 0, 255]),
        82:
        dict(link=('sso_kpt21', 'sso_kpt20'), id=82, color=[128, 0, 255]),
        83:
        dict(link=('sso_kpt20', 'sso_kpt19'), id=83, color=[128, 0, 255]),
        84:
        dict(link=('sso_kpt19', 'sso_kpt18'), id=84, color=[128, 0, 255]),
        85:
        dict(link=('sso_kpt18', 'sso_kpt17'), id=85, color=[128, 0, 255]),
        86:
        dict(link=('sso_kpt17', 'sso_kpt29'), id=86, color=[128, 0, 255]),
        87:
        dict(link=('sso_kpt29', 'sso_kpt28'), id=87, color=[128, 0, 255]),
        88:
        dict(link=('sso_kpt28', 'sso_kpt27'), id=88, color=[128, 0, 255]),
        89:
        dict(link=('sso_kpt27', 'sso_kpt26'), id=89, color=[128, 0, 255]),
        90:
        dict(link=('sso_kpt26', 'sso_kpt5'), id=90, color=[128, 0, 255]),
        91:
        dict(link=('sso_kpt5', 'sso_kpt6'), id=91, color=[128, 0, 255]),
        # long_sleeved_outwear
        92:
        dict(link=('lso_kpt1', 'lso_kpt2'), id=92, color=[0, 128, 255]),
        93:
        dict(link=('lso_kpt2', 'lso_kpt7'), id=93, color=[0, 128, 255]),
        94:
        dict(link=('lso_kpt7', 'lso_kpt8'), id=94, color=[0, 128, 255]),
        95:
        dict(link=('lso_kpt8', 'lso_kpt9'), id=95, color=[0, 128, 255]),
        96:
        dict(link=('lso_kpt9', 'lso_kpt10'), id=96, color=[0, 128, 255]),
        97:
        dict(link=('lso_kpt10', 'lso_kpt11'), id=97, color=[0, 128, 255]),
        98:
        dict(link=('lso_kpt11', 'lso_kpt12'), id=98, color=[0, 128, 255]),
        99:
        dict(link=('lso_kpt12', 'lso_kpt13'), id=99, color=[0, 128, 255]),
        100:
        dict(link=('lso_kpt13', 'lso_kpt14'), id=100, color=[0, 128, 255]),
        101:
        dict(link=('lso_kpt14', 'lso_kpt15'), id=101, color=[0, 128, 255]),
        102:
        dict(link=('lso_kpt15', 'lso_kpt16'), id=102, color=[0, 128, 255]),
        103:
        dict(link=('lso_kpt16', 'lso_kpt17'), id=103, color=[0, 128, 255]),
        104:
        dict(link=('lso_kpt17', 'lso_kpt18'), id=104, color=[0, 128, 255]),
        105:
        dict(link=('lso_kpt18', 'lso_kpt19'), id=105, color=[0, 128, 255]),
        106:
        dict(link=('lso_kpt19', 'lso_kpt20'), id=106, color=[0, 128, 255]),
        107:
        dict(link=('lso_kpt20', 'lso_kpt39'), id=107, color=[0, 128, 255]),
        108:
        dict(link=('lso_kpt39', 'lso_kpt38'), id=108, color=[0, 128, 255]),
        109:
        dict(link=('lso_kpt38', 'lso_kpt4'), id=109, color=[0, 128, 255]),
        110:
        dict(link=('lso_kpt4', 'lso_kpt3'), id=110, color=[0, 128, 255]),
        111:
        dict(link=('lso_kpt3', 'lso_kpt2'), id=111, color=[0, 128, 255]),
        112:
        dict(link=('lso_kpt1', 'lso_kpt6'), id=112, color=[0, 128, 255]),
        113:
        dict(link=('lso_kpt6', 'lso_kpt33'), id=113, color=[0, 128, 255]),
        114:
        dict(link=('lso_kpt33', 'lso_kpt32'), id=114, color=[0, 128, 255]),
        115:
        dict(link=('lso_kpt32', 'lso_kpt31'), id=115, color=[0, 128, 255]),
        116:
        dict(link=('lso_kpt31', 'lso_kpt30'), id=116, color=[0, 128, 255]),
        117:
        dict(link=('lso_kpt30', 'lso_kpt29'), id=117, color=[0, 128, 255]),
        118:
        dict(link=('lso_kpt29', 'lso_kpt28'), id=118, color=[0, 128, 255]),
        119:
        dict(link=('lso_kpt28', 'lso_kpt27'), id=119, color=[0, 128, 255]),
        120:
        dict(link=('lso_kpt27', 'lso_kpt26'), id=120, color=[0, 128, 255]),
        121:
        dict(link=('lso_kpt26', 'lso_kpt25'), id=121, color=[0, 128, 255]),
        122:
        dict(link=('lso_kpt25', 'lso_kpt24'), id=122, color=[0, 128, 255]),
        123:
        dict(link=('lso_kpt24', 'lso_kpt23'), id=123, color=[0, 128, 255]),
        124:
        dict(link=('lso_kpt23', 'lso_kpt22'), id=124, color=[0, 128, 255]),
        125:
        dict(link=('lso_kpt22', 'lso_kpt21'), id=125, color=[0, 128, 255]),
        126:
        dict(link=('lso_kpt21', 'lso_kpt37'), id=126, color=[0, 128, 255]),
        127:
        dict(link=('lso_kpt37', 'lso_kpt36'), id=127, color=[0, 128, 255]),
        128:
        dict(link=('lso_kpt36', 'lso_kpt35'), id=128, color=[0, 128, 255]),
        129:
        dict(link=('lso_kpt35', 'lso_kpt34'), id=129, color=[0, 128, 255]),
        130:
        dict(link=('lso_kpt34', 'lso_kpt5'), id=130, color=[0, 128, 255]),
        131:
        dict(link=('lso_kpt5', 'lso_kpt6'), id=131, color=[0, 128, 255]),
        # vest
        132:
        dict(link=('vest_kpt1', 'vest_kpt2'), id=132, color=[0, 128, 128]),
        133:
        dict(link=('vest_kpt2', 'vest_kpt7'), id=133, color=[0, 128, 128]),
        134:
        dict(link=('vest_kpt7', 'vest_kpt8'), id=134, color=[0, 128, 128]),
        135:
        dict(link=('vest_kpt8', 'vest_kpt9'), id=135, color=[0, 128, 128]),
        136:
        dict(link=('vest_kpt9', 'vest_kpt10'), id=136, color=[0, 128, 128]),
        137:
        dict(link=('vest_kpt10', 'vest_kpt11'), id=137, color=[0, 128, 128]),
        138:
        dict(link=('vest_kpt11', 'vest_kpt12'), id=138, color=[0, 128, 128]),
        139:
        dict(link=('vest_kpt12', 'vest_kpt13'), id=139, color=[0, 128, 128]),
        140:
        dict(link=('vest_kpt13', 'vest_kpt14'), id=140, color=[0, 128, 128]),
        141:
        dict(link=('vest_kpt14', 'vest_kpt15'), id=141, color=[0, 128, 128]),
        142:
        dict(link=('vest_kpt15', 'vest_kpt6'), id=142, color=[0, 128, 128]),
        143:
        dict(link=('vest_kpt6', 'vest_kpt1'), id=143, color=[0, 128, 128]),
        144:
        dict(link=('vest_kpt2', 'vest_kpt3'), id=144, color=[0, 128, 128]),
        145:
        dict(link=('vest_kpt3', 'vest_kpt4'), id=145, color=[0, 128, 128]),
        146:
        dict(link=('vest_kpt4', 'vest_kpt5'), id=146, color=[0, 128, 128]),
        147:
        dict(link=('vest_kpt5', 'vest_kpt6'), id=147, color=[0, 128, 128]),
        # sling
        148:
        dict(link=('sling_kpt1', 'sling_kpt2'), id=148, color=[0, 0, 128]),
        149:
        dict(link=('sling_kpt2', 'sling_kpt8'), id=149, color=[0, 0, 128]),
        150:
        dict(link=('sling_kpt8', 'sling_kpt9'), id=150, color=[0, 0, 128]),
        151:
        dict(link=('sling_kpt9', 'sling_kpt10'), id=151, color=[0, 0, 128]),
        152:
        dict(link=('sling_kpt10', 'sling_kpt11'), id=152, color=[0, 0, 128]),
        153:
        dict(link=('sling_kpt11', 'sling_kpt12'), id=153, color=[0, 0, 128]),
        154:
        dict(link=('sling_kpt12', 'sling_kpt13'), id=154, color=[0, 0, 128]),
        155:
        dict(link=('sling_kpt13', 'sling_kpt14'), id=155, color=[0, 0, 128]),
        156:
        dict(link=('sling_kpt14', 'sling_kpt6'), id=156, color=[0, 0, 128]),
        157:
        dict(link=('sling_kpt2', 'sling_kpt7'), id=157, color=[0, 0, 128]),
        158:
        dict(link=('sling_kpt6', 'sling_kpt15'), id=158, color=[0, 0, 128]),
        159:
        dict(link=('sling_kpt2', 'sling_kpt3'), id=159, color=[0, 0, 128]),
        160:
        dict(link=('sling_kpt3', 'sling_kpt4'), id=160, color=[0, 0, 128]),
        161:
        dict(link=('sling_kpt4', 'sling_kpt5'), id=161, color=[0, 0, 128]),
        162:
        dict(link=('sling_kpt5', 'sling_kpt6'), id=162, color=[0, 0, 128]),
        163:
        dict(link=('sling_kpt1', 'sling_kpt6'), id=163, color=[0, 0, 128]),
        # shorts
        164:
        dict(
            link=('shorts_kpt1', 'shorts_kpt4'), id=164, color=[128, 128,
                                                                128]),
        165:
        dict(
            link=('shorts_kpt4', 'shorts_kpt5'), id=165, color=[128, 128,
                                                                128]),
        166:
        dict(
            link=('shorts_kpt5', 'shorts_kpt6'), id=166, color=[128, 128,
                                                                128]),
        167:
        dict(
            link=('shorts_kpt6', 'shorts_kpt7'), id=167, color=[128, 128,
                                                                128]),
        168:
        dict(
            link=('shorts_kpt7', 'shorts_kpt8'), id=168, color=[128, 128,
                                                                128]),
        169:
        dict(
            link=('shorts_kpt8', 'shorts_kpt9'), id=169, color=[128, 128,
                                                                128]),
        170:
        dict(
            link=('shorts_kpt9', 'shorts_kpt10'),
            id=170,
            color=[128, 128, 128]),
        171:
        dict(
            link=('shorts_kpt10', 'shorts_kpt3'),
            id=171,
            color=[128, 128, 128]),
        172:
        dict(
            link=('shorts_kpt3', 'shorts_kpt2'), id=172, color=[128, 128,
                                                                128]),
        173:
        dict(
            link=('shorts_kpt2', 'shorts_kpt1'), id=173, color=[128, 128,
                                                                128]),
        # trousers
        174:
        dict(
            link=('trousers_kpt1', 'trousers_kpt4'),
            id=174,
            color=[128, 0, 128]),
        175:
        dict(
            link=('trousers_kpt4', 'trousers_kpt5'),
            id=175,
            color=[128, 0, 128]),
        176:
        dict(
            link=('trousers_kpt5', 'trousers_kpt6'),
            id=176,
            color=[128, 0, 128]),
        177:
        dict(
            link=('trousers_kpt6', 'trousers_kpt7'),
            id=177,
            color=[128, 0, 128]),
        178:
        dict(
            link=('trousers_kpt7', 'trousers_kpt8'),
            id=178,
            color=[128, 0, 128]),
        179:
        dict(
            link=('trousers_kpt8', 'trousers_kpt9'),
            id=179,
            color=[128, 0, 128]),
        180:
        dict(
            link=('trousers_kpt9', 'trousers_kpt10'),
            id=180,
            color=[128, 0, 128]),
        181:
        dict(
            link=('trousers_kpt10', 'trousers_kpt11'),
            id=181,
            color=[128, 0, 128]),
        182:
        dict(
            link=('trousers_kpt11', 'trousers_kpt12'),
            id=182,
            color=[128, 0, 128]),
        183:
        dict(
            link=('trousers_kpt12', 'trousers_kpt13'),
            id=183,
            color=[128, 0, 128]),
        184:
        dict(
            link=('trousers_kpt13', 'trousers_kpt14'),
            id=184,
            color=[128, 0, 128]),
        185:
        dict(
            link=('trousers_kpt14', 'trousers_kpt3'),
            id=185,
            color=[128, 0, 128]),
        186:
        dict(
            link=('trousers_kpt3', 'trousers_kpt2'),
            id=186,
            color=[128, 0, 128]),
        187:
        dict(
            link=('trousers_kpt2', 'trousers_kpt1'),
            id=187,
            color=[128, 0, 128]),
        # skirt
        188:
        dict(link=('skirt_kpt1', 'skirt_kpt4'), id=188, color=[64, 128, 128]),
        189:
        dict(link=('skirt_kpt4', 'skirt_kpt5'), id=189, color=[64, 128, 128]),
        190:
        dict(link=('skirt_kpt5', 'skirt_kpt6'), id=190, color=[64, 128, 128]),
        191:
        dict(link=('skirt_kpt6', 'skirt_kpt7'), id=191, color=[64, 128, 128]),
        192:
        dict(link=('skirt_kpt7', 'skirt_kpt8'), id=192, color=[64, 128, 128]),
        193:
        dict(link=('skirt_kpt8', 'skirt_kpt3'), id=193, color=[64, 128, 128]),
        194:
        dict(link=('skirt_kpt3', 'skirt_kpt2'), id=194, color=[64, 128, 128]),
        195:
        dict(link=('skirt_kpt2', 'skirt_kpt1'), id=195, color=[64, 128, 128]),
        # short_sleeved_dress
        196:
        dict(link=('ssd_kpt1', 'ssd_kpt2'), id=196, color=[64, 64, 128]),
        197:
        dict(link=('ssd_kpt2', 'ssd_kpt7'), id=197, color=[64, 64, 128]),
        198:
        dict(link=('ssd_kpt7', 'ssd_kpt8'), id=198, color=[64, 64, 128]),
        199:
        dict(link=('ssd_kpt8', 'ssd_kpt9'), id=199, color=[64, 64, 128]),
        200:
        dict(link=('ssd_kpt9', 'ssd_kpt10'), id=200, color=[64, 64, 128]),
        201:
        dict(link=('ssd_kpt10', 'ssd_kpt11'), id=201, color=[64, 64, 128]),
        202:
        dict(link=('ssd_kpt11', 'ssd_kpt12'), id=202, color=[64, 64, 128]),
        203:
        dict(link=('ssd_kpt12', 'ssd_kpt13'), id=203, color=[64, 64, 128]),
        204:
        dict(link=('ssd_kpt13', 'ssd_kpt14'), id=204, color=[64, 64, 128]),
        205:
        dict(link=('ssd_kpt14', 'ssd_kpt15'), id=205, color=[64, 64, 128]),
        206:
        dict(link=('ssd_kpt15', 'ssd_kpt16'), id=206, color=[64, 64, 128]),
        207:
        dict(link=('ssd_kpt16', 'ssd_kpt17'), id=207, color=[64, 64, 128]),
        208:
        dict(link=('ssd_kpt17', 'ssd_kpt18'), id=208, color=[64, 64, 128]),
        209:
        dict(link=('ssd_kpt18', 'ssd_kpt19'), id=209, color=[64, 64, 128]),
        210:
        dict(link=('ssd_kpt19', 'ssd_kpt20'), id=210, color=[64, 64, 128]),
        211:
        dict(link=('ssd_kpt20', 'ssd_kpt21'), id=211, color=[64, 64, 128]),
        212:
        dict(link=('ssd_kpt21', 'ssd_kpt22'), id=212, color=[64, 64, 128]),
        213:
        dict(link=('ssd_kpt22', 'ssd_kpt23'), id=213, color=[64, 64, 128]),
        214:
        dict(link=('ssd_kpt23', 'ssd_kpt24'), id=214, color=[64, 64, 128]),
        215:
        dict(link=('ssd_kpt24', 'ssd_kpt25'), id=215, color=[64, 64, 128]),
        216:
        dict(link=('ssd_kpt25', 'ssd_kpt26'), id=216, color=[64, 64, 128]),
        217:
        dict(link=('ssd_kpt26', 'ssd_kpt27'), id=217, color=[64, 64, 128]),
        218:
        dict(link=('ssd_kpt27', 'ssd_kpt28'), id=218, color=[64, 64, 128]),
        219:
        dict(link=('ssd_kpt28', 'ssd_kpt29'), id=219, color=[64, 64, 128]),
        220:
        dict(link=('ssd_kpt29', 'ssd_kpt6'), id=220, color=[64, 64, 128]),
        221:
        dict(link=('ssd_kpt6', 'ssd_kpt5'), id=221, color=[64, 64, 128]),
        222:
        dict(link=('ssd_kpt5', 'ssd_kpt4'), id=222, color=[64, 64, 128]),
        223:
        dict(link=('ssd_kpt4', 'ssd_kpt3'), id=223, color=[64, 64, 128]),
        224:
        dict(link=('ssd_kpt3', 'ssd_kpt2'), id=224, color=[64, 64, 128]),
        225:
        dict(link=('ssd_kpt6', 'ssd_kpt1'), id=225, color=[64, 64, 128]),
        # long_sleeved_dress
        226:
        dict(link=('lsd_kpt1', 'lsd_kpt2'), id=226, color=[128, 64, 0]),
        227:
        dict(link=('lsd_kpt2', 'lsd_kpt7'), id=228, color=[128, 64, 0]),
        228:
        dict(link=('lsd_kpt7', 'lsd_kpt8'), id=228, color=[128, 64, 0]),
        229:
        dict(link=('lsd_kpt8', 'lsd_kpt9'), id=229, color=[128, 64, 0]),
        230:
        dict(link=('lsd_kpt9', 'lsd_kpt10'), id=230, color=[128, 64, 0]),
        231:
        dict(link=('lsd_kpt10', 'lsd_kpt11'), id=231, color=[128, 64, 0]),
        232:
        dict(link=('lsd_kpt11', 'lsd_kpt12'), id=232, color=[128, 64, 0]),
        233:
        dict(link=('lsd_kpt12', 'lsd_kpt13'), id=233, color=[128, 64, 0]),
        234:
        dict(link=('lsd_kpt13', 'lsd_kpt14'), id=234, color=[128, 64, 0]),
        235:
        dict(link=('lsd_kpt14', 'lsd_kpt15'), id=235, color=[128, 64, 0]),
        236:
        dict(link=('lsd_kpt15', 'lsd_kpt16'), id=236, color=[128, 64, 0]),
        237:
        dict(link=('lsd_kpt16', 'lsd_kpt17'), id=237, color=[128, 64, 0]),
        238:
        dict(link=('lsd_kpt17', 'lsd_kpt18'), id=238, color=[128, 64, 0]),
        239:
        dict(link=('lsd_kpt18', 'lsd_kpt19'), id=239, color=[128, 64, 0]),
        240:
        dict(link=('lsd_kpt19', 'lsd_kpt20'), id=240, color=[128, 64, 0]),
        241:
        dict(link=('lsd_kpt20', 'lsd_kpt21'), id=241, color=[128, 64, 0]),
        242:
        dict(link=('lsd_kpt21', 'lsd_kpt22'), id=242, color=[128, 64, 0]),
        243:
        dict(link=('lsd_kpt22', 'lsd_kpt23'), id=243, color=[128, 64, 0]),
        244:
        dict(link=('lsd_kpt23', 'lsd_kpt24'), id=244, color=[128, 64, 0]),
        245:
        dict(link=('lsd_kpt24', 'lsd_kpt25'), id=245, color=[128, 64, 0]),
        246:
        dict(link=('lsd_kpt25', 'lsd_kpt26'), id=246, color=[128, 64, 0]),
        247:
        dict(link=('lsd_kpt26', 'lsd_kpt27'), id=247, color=[128, 64, 0]),
        248:
        dict(link=('lsd_kpt27', 'lsd_kpt28'), id=248, color=[128, 64, 0]),
        249:
        dict(link=('lsd_kpt28', 'lsd_kpt29'), id=249, color=[128, 64, 0]),
        250:
        dict(link=('lsd_kpt29', 'lsd_kpt30'), id=250, color=[128, 64, 0]),
        251:
        dict(link=('lsd_kpt30', 'lsd_kpt31'), id=251, color=[128, 64, 0]),
        252:
        dict(link=('lsd_kpt31', 'lsd_kpt32'), id=252, color=[128, 64, 0]),
        253:
        dict(link=('lsd_kpt32', 'lsd_kpt33'), id=253, color=[128, 64, 0]),
        254:
        dict(link=('lsd_kpt33', 'lsd_kpt34'), id=254, color=[128, 64, 0]),
        255:
        dict(link=('lsd_kpt34', 'lsd_kpt35'), id=255, color=[128, 64, 0]),
        256:
        dict(link=('lsd_kpt35', 'lsd_kpt36'), id=256, color=[128, 64, 0]),
        257:
        dict(link=('lsd_kpt36', 'lsd_kpt37'), id=257, color=[128, 64, 0]),
        258:
        dict(link=('lsd_kpt37', 'lsd_kpt6'), id=258, color=[128, 64, 0]),
        259:
        dict(link=('lsd_kpt6', 'lsd_kpt5'), id=259, color=[128, 64, 0]),
        260:
        dict(link=('lsd_kpt5', 'lsd_kpt4'), id=260, color=[128, 64, 0]),
        261:
        dict(link=('lsd_kpt4', 'lsd_kpt3'), id=261, color=[128, 64, 0]),
        262:
        dict(link=('lsd_kpt3', 'lsd_kpt2'), id=262, color=[128, 64, 0]),
        263:
        dict(link=('lsd_kpt6', 'lsd_kpt1'), id=263, color=[128, 64, 0]),
        # vest_dress
        264:
        dict(link=('vd_kpt1', 'vd_kpt2'), id=264, color=[128, 64, 255]),
        265:
        dict(link=('vd_kpt2', 'vd_kpt7'), id=265, color=[128, 64, 255]),
        266:
        dict(link=('vd_kpt7', 'vd_kpt8'), id=266, color=[128, 64, 255]),
        267:
        dict(link=('vd_kpt8', 'vd_kpt9'), id=267, color=[128, 64, 255]),
        268:
        dict(link=('vd_kpt9', 'vd_kpt10'), id=268, color=[128, 64, 255]),
        269:
        dict(link=('vd_kpt10', 'vd_kpt11'), id=269, color=[128, 64, 255]),
        270:
        dict(link=('vd_kpt11', 'vd_kpt12'), id=270, color=[128, 64, 255]),
        271:
        dict(link=('vd_kpt12', 'vd_kpt13'), id=271, color=[128, 64, 255]),
        272:
        dict(link=('vd_kpt13', 'vd_kpt14'), id=272, color=[128, 64, 255]),
        273:
        dict(link=('vd_kpt14', 'vd_kpt15'), id=273, color=[128, 64, 255]),
        274:
        dict(link=('vd_kpt15', 'vd_kpt16'), id=274, color=[128, 64, 255]),
        275:
        dict(link=('vd_kpt16', 'vd_kpt17'), id=275, color=[128, 64, 255]),
        276:
        dict(link=('vd_kpt17', 'vd_kpt18'), id=276, color=[128, 64, 255]),
        277:
        dict(link=('vd_kpt18', 'vd_kpt19'), id=277, color=[128, 64, 255]),
        278:
        dict(link=('vd_kpt19', 'vd_kpt6'), id=278, color=[128, 64, 255]),
        279:
        dict(link=('vd_kpt6', 'vd_kpt5'), id=279, color=[128, 64, 255]),
        280:
        dict(link=('vd_kpt5', 'vd_kpt4'), id=280, color=[128, 64, 255]),
        281:
        dict(link=('vd_kpt4', 'vd_kpt3'), id=281, color=[128, 64, 255]),
        282:
        dict(link=('vd_kpt3', 'vd_kpt2'), id=282, color=[128, 64, 255]),
        283:
        dict(link=('vd_kpt6', 'vd_kpt1'), id=283, color=[128, 64, 255]),
        # sling_dress
        284:
        dict(link=('sd_kpt1', 'sd_kpt2'), id=284, color=[128, 64, 0]),
        285:
        dict(link=('sd_kpt2', 'sd_kpt8'), id=285, color=[128, 64, 0]),
        286:
        dict(link=('sd_kpt8', 'sd_kpt9'), id=286, color=[128, 64, 0]),
        287:
        dict(link=('sd_kpt9', 'sd_kpt10'), id=287, color=[128, 64, 0]),
        288:
        dict(link=('sd_kpt10', 'sd_kpt11'), id=288, color=[128, 64, 0]),
        289:
        dict(link=('sd_kpt11', 'sd_kpt12'), id=289, color=[128, 64, 0]),
        290:
        dict(link=('sd_kpt12', 'sd_kpt13'), id=290, color=[128, 64, 0]),
        291:
        dict(link=('sd_kpt13', 'sd_kpt14'), id=291, color=[128, 64, 0]),
        292:
        dict(link=('sd_kpt14', 'sd_kpt15'), id=292, color=[128, 64, 0]),
        293:
        dict(link=('sd_kpt15', 'sd_kpt16'), id=293, color=[128, 64, 0]),
        294:
        dict(link=('sd_kpt16', 'sd_kpt17'), id=294, color=[128, 64, 0]),
        295:
        dict(link=('sd_kpt17', 'sd_kpt18'), id=295, color=[128, 64, 0]),
        296:
        dict(link=('sd_kpt18', 'sd_kpt6'), id=296, color=[128, 64, 0]),
        297:
        dict(link=('sd_kpt6', 'sd_kpt5'), id=297, color=[128, 64, 0]),
        298:
        dict(link=('sd_kpt5', 'sd_kpt4'), id=298, color=[128, 64, 0]),
        299:
        dict(link=('sd_kpt4', 'sd_kpt3'), id=299, color=[128, 64, 0]),
        300:
        dict(link=('sd_kpt3', 'sd_kpt2'), id=300, color=[128, 64, 0]),
        301:
        dict(link=('sd_kpt2', 'sd_kpt7'), id=301, color=[128, 64, 0]),
        302:
        dict(link=('sd_kpt6', 'sd_kpt19'), id=302, color=[128, 64, 0]),
        303:
        dict(link=('sd_kpt6', 'sd_kpt1'), id=303, color=[128, 64, 0]),
    },
    joint_weights=[1.] * 294,
    sigmas=[])
