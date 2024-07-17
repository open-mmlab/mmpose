# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# import sys


class Labelme2coco_keypoints():

    def __init__(self, args):
        """Lableme 关键点数据集转 COCO 数据集的构造函数:

        Args
            args：命令行输入的参数
                - class_name 根类名字
        """

        self.classname_to_id = {args.class_name: 1}
        self.images = []
        self.annotations = []
        self.categories = []
        self.ann_id = 0
        self.img_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(
            instance,
            open(save_path, 'w', encoding='utf-8'),
            ensure_ascii=False,
            indent=1)

    def read_jsonfile(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_keypoints(self, points, keypoints, num_keypoints, label):
        """解析 labelme 的原始数据， 生成 coco 标注的 关键点对象.

        例如：
            "keypoints": [
                67.06149888292556,  # x 的值
                122.5043507571318,  # y 的值
                1,                  # 相当于 Z 值，2D关键点 v = 0表示不可见,
                                      v = 1表示标记但不可见，v = 2表示标记且可见
                82.42582269256718,
                109.95672933232304,
                1,
                ...,
            ],
        """
        labels = ['wrist', 'thumb1', 'thumb2', ...]
        flag = label.split('_')[-1]
        x = label.split('_')[0]
        visible = 0
        if flag == 'occluded':
            visible = 1
        else:
            visible = 2
        x = labels.index(x)
        keypoints[x * 3] = points[0]
        keypoints[x * 3 + 1] = points[1]
        keypoints[x * 3 + 2] = visible
        num_keypoints += 1

        return num_keypoints

    def _image(self, obj, path):
        """解析 labelme 的 obj 对象，生成 coco 的 image 对象.

        生成包括：id，file_name，height，width 4个属性

        示例：
             {
                "file_name": "training/rgb/00031426.jpg",
                "height": 224,
                "width": 224,
                "id": 31426
            }
        """

        image = {}

        # 此处通过imageData获得数据
        # 获得原始 labelme 标签的 imageData 属性，并通过 labelme 的工具方法转成 array
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # image['height'], image['width'] = img_x.shape[:-1]  # 获得图片的宽高

        # 此处直接通过imageHeight，imageWidth得到,避免labelme中的imageData问题
        image['height'], image['width'] = obj['imageHeight'], obj[
            'imageWidth']  # 获得图片的宽高
        # self.img_id = int(os.path.basename(path).split(".json")[0])
        self.img_id = self.img_id + 1
        image['id'] = self.img_id

        image['file_name'] = os.path.basename(path).replace('.json', '.jpg')

        return image

    def _annotation(self, bboxes_list, keypoints_list, json_path):
        """生成coco标注.

        Args：     bboxes_list： 矩形标注框     keypoints_list： 关键点 json_path：json文件路径
        """
        # 核对一个bbox里有n个keypoints； 然而本人不要求每个bbox里都要有n个点
        # if len(keypoints_list) != args.join_num * len(bboxes_list):
        #     print(
        #         'you loss {} keypoint(s) with file {}'\
        #         .format(args.join_num * len(bboxes_list) -\
        #         len(keypoints_list), json_path)
        #     )
        #     print('Please check ！！！')
        #     sys.exit()

        i = 0
        # 对每个bbox分别保存keypoints
        for object in bboxes_list:
            annotation = {}
            keypoints = [0 for i in range(36)
                         ]  # 每个keypoint数组初始化为[0,..] len = 36 对应12个点(x,y,v)
            num_keypoints = 0

            label = object['label']
            bbox = object['points']
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = int(self.classname_to_id[label])
            annotation['iscrowd'] = 0
            annotation['area'] = 1.0
            annotation['segmentation'] = [np.asarray(bbox).flatten().tolist()
                                          ]  # 两个坐标点
            annotation['bbox'] = self._get_box(bbox)  # 矩形框左上角的坐标和矩形框的长宽

            # 生成keypoint的list
            for keypoint in keypoints_list:
                point = keypoint['points']
                label = keypoint['label']  # 点的名字
                num_keypoints = self._get_keypoints(point[0], keypoints,
                                                    num_keypoints, label)
            annotation['keypoints'] = keypoints
            annotation['num_keypoints'] = num_keypoints

            i += 1
            self.ann_id += 1
            self.annotations.append(annotation)

    def _init_categories(self):
        """初始化 COCO 的 标注类别.

        例如：
        "categories": [
            {
                "supercategory": "hand",
                "id": 1,
                "name": "hand",
                "keypoints": [
                    "wrist",
                    "thumb1",
                    "thumb2",
                    ...,
                ],
                "skeleton": [
                ]
            }
        ]
        """

        for name, id in self.classname_to_id.items():
            category = {}

            category['supercategory'] = name
            category['id'] = id
            category['name'] = name
            # n个关键点数据
            category['keypoint'] = [
                'wrist',
                'thumb1',
                'thumb2',
                ...,
            ]
            # category['keypoint'] = [str(i + 1) for i in range(args.join_num)]

            self.categories.append(category)

    def to_coco(self, json_path_list):
        """Labelme 原始标签转换成 coco 数据集格式，生成的包括标签和图像.

        Args：     json_path_list：原始数据集的目录
        """

        self._init_categories()
        # 整个文件夹里的json进行逐个处理
        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)  # 解析一个标注文件
            self.images.append(self._image(obj, json_path))  # 解析图片
            shapes = obj['shapes']  # 读取 labelme shape 标注

            bboxes_list, keypoints_list = [], []
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':  # bboxs
                    bboxes_list.append(shape)
                elif shape['shape_type'] == 'point':  # keypoints
                    keypoints_list.append(shape)
            # 输入为一个文件的keypoints和bbox，即一张图里的信息
            self._annotation(bboxes_list, keypoints_list, json_path)

        keypoints = {}
        keypoints['info'] = {
            'description': 'Air Dataset',
            'version': 1.0,
            'year': 2022
        }
        keypoints['license'] = ['BUAA']
        keypoints['images'] = self.images
        keypoints['annotations'] = self.annotations
        keypoints['categories'] = self.categories
        return keypoints


def init_dir(base_path):
    """初始化COCO数据集的文件夹结构；

    coco - annotations  #标注文件路径
         - train        #训练数据集
         - val          #验证数据集
    Args：
        base_path：数据集放置的根路径
    """
    if not os.path.exists(os.path.join(base_path, 'coco', 'annotations')):
        os.makedirs(os.path.join(base_path, 'coco', 'annotations'))
    if not os.path.exists(os.path.join(base_path, 'coco', 'train')):
        os.makedirs(os.path.join(base_path, 'coco', 'train'))
    if not os.path.exists(os.path.join(base_path, 'coco', 'val')):
        os.makedirs(os.path.join(base_path, 'coco', 'val'))


def convert(path, target):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--class_name', '--n', help='class name', type=str, default='airplane')
    parser.add_argument(
        '--input',
        '--i',
        help='json file path (labelme)',
        type=str,
        default=path)
    parser.add_argument(
        '--output',
        '--o',
        help='output file path (coco format)',
        type=str,
        default=path)
    parser.add_argument(
        '--join_num', '--j', help='number of join', type=int, default=12)
    parser.add_argument(
        '--ratio',
        '--r',
        help='train and test split ratio',
        type=float,
        default=0.25)
    args = parser.parse_args()

    labelme_path = args.input
    saved_coco_path = args.output

    init_dir(saved_coco_path)  # 初始化COCO数据集的文件夹结构

    json_list_path = glob.glob(labelme_path + '/*.json')
    train_path, val_path = train_test_split(
        json_list_path, test_size=args.ratio)
    print('{} for training'.format(len(train_path)),
          '\n{} for testing'.format(len(val_path)))
    print('Start transform please wait ...')

    l2c_train = Labelme2coco_keypoints(args)  # 构造数据集生成类

    # 生成训练集
    train_keypoints = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(
        train_keypoints,
        os.path.join(saved_coco_path, 'coco', 'annotations',
                     'keypoints_train.json'))

    # 生成验证集
    l2c_val = Labelme2coco_keypoints(args)
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(
        val_instance,
        os.path.join(saved_coco_path, 'coco', 'annotations',
                     'keypoints_val.json'))

    # 拷贝 labelme 的原始图片到训练集和验证集里面
    for file in train_path:
        shutil.copy(
            file.replace('json', 'jpg'),
            os.path.join(saved_coco_path, 'coco', 'train'))
    for file in val_path:
        shutil.copy(
            file.replace('json', 'jpg'),
            os.path.join(saved_coco_path, 'coco', 'val'))


if __name__ == '__main__':
    source = 'your labelme path'
    target = 'your coco path'
    convert(source, target)
