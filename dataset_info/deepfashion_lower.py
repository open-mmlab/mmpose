dataset_info = dict(
    dataset_name='deepfashion_lower',
    paper_info=dict(
        author='Liu, Ziwei and Luo, Ping and Qiu, Shi '
        'and Wang, Xiaogang and Tang, Xiaoou',
        title='DeepFashion: Powering Robust Clothes Recognition '
        'and Retrieval with Rich Annotations',
        container='Proceedings of IEEE Conference on Computer '
        'Vision and Pattern Recognition (CVPR)',
        year='2016',
        homepage='http://mmlab.ie.cuhk.edu.hk/projects/'
        'DeepFashion/LandmarkDetection.html',
    ),
    keypoint_info={
        0:
        dict(
            name='left waistline',
            id=0,
            color=[255, 255, 255],
            type='',
            swap='right waistline'),
        1:
        dict(
            name='right waistline',
            id=1,
            color=[255, 255, 255],
            type='',
            swap='left waistline'),
        2:
        dict(
            name='left hem',
            id=2,
            color=[255, 255, 255],
            type='',
            swap='right hem'),
        3:
        dict(
            name='right hem',
            id=3,
            color=[255, 255, 255],
            type='',
            swap='left hem'),
    },
    skeleton_info={
        0:
        dict(
            link=('left waistline', 'right waistline'),
            id=0,
            color=[255, 255, 255]),
        1:
        dict(
            link=('right waistline', 'right hem'), id=1, color=[255, 255,
                                                                255]),
        2:
        dict(link=('right hem', 'left hem'), id=2, color=[255, 255, 255]),
        3:
        dict(link=('left hem', 'left waistline'), id=3, color=[255, 255, 255]),
    },
    joint_weights=[1.] * 4,
    sigmas=[])
