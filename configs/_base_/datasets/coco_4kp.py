# Scaletronic 4kp
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(
            name='top_left', 
            id=0, 
            color=[51, 153, 255], 
            type='upper', 
            swap='top_right'),
        1:
        dict(
            name='top_right',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='top_left'),
        2:
        dict(
            name='bottom_right',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='bottom_left'),
        3:
        dict(
            name='bottom_left',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='bottom_right')
    },
    skeleton_info={
        0:
        dict(link=('top_left', 'top_right'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('bottom_left', 'bottom_right'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('top_left', 'bottom_left'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('top_right', 'bottom_right'), id=3, color=[0, 255, 0])
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[
        0.05, 0.05, 0.05, 0.05
    ])