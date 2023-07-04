# -----------------------------------------------------------------------------
# Based on https://github.com/heartexlabs/label-studio-converter
# Original license: Copyright (c) Heartex, under the Apache 2.0 License.
# -----------------------------------------------------------------------------

import argparse
import io
import json
import logging
import pathlib
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio JSON file to COCO format JSON File')
    parser.add_argument('config', help='Labeling Interface xml code file path')
    parser.add_argument('input', help='Label Studio format JSON file path')
    parser.add_argument('output', help='The output COCO format JSON file path')
    args = parser.parse_args()
    return args


class LSConverter:

    def __init__(self, config: str):
        """Convert the Label Studio Format JSON file to COCO format JSON file
        which is needed by mmpose.

           The annotations in label studio must follow the order:
           keypoint 1, keypoint 2... keypoint n, rect of the instance,
           polygon of the instance,
           then annotations of the next instance.
           Where the order of rect and polygon can be switched,
           the bbox and area of the instance will be calculated with
           the data behind.

           Only annotating one of rect and polygon is also acceptable.
        Args:
            config (str): The annotations config xml file.
                The xml content is from Project Setting ->
                Label Interface -> Code.
                Example:
                ```
                <View>
                <KeyPointLabels name="kp-1" toName="img-1">
                    <Label value="person" background="#D4380D"/>
                </KeyPointLabels>
                <PolygonLabels name="polygonlabel" toName="img-1">
                    <Label value="person" background="#0DA39E"/>
                </PolygonLabels>
                <RectangleLabels name="label" toName="img-1">
                    <Label value="person" background="#DDA0EE"/>
                </RectangleLabels>
                <Image name="img-1" value="$img"/>
                </View>
                ```
        """
        # get label info from config file
        tree = ET.parse(config)
        root = tree.getroot()
        labels = root.findall('.//KeyPointLabels/Label')
        label_values = [label.get('value') for label in labels]

        self.categories = list()
        self.category_name_to_id = dict()
        for i, value in enumerate(label_values):
            # category id start with 1
            self.categories.append({'id': i + 1, 'name': value})
            self.category_name_to_id[value] = i + 1

    def convert_to_coco(self, input_json: str, output_json: str):
        """Convert `input_json` to COCO format and save in `output_json`.

        Args:
            input_json (str): The path of Label Studio format JSON file.
            output_json (str): The path of the output COCO JSON file.
        """

        def add_image(images, width, height, image_id, image_path):
            images.append({
                'width': width,
                'height': height,
                'id': image_id,
                'file_name': image_path,
            })
            return images

        output_path = pathlib.Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        images = list()
        annotations = list()

        with open(input_json, 'r') as f:
            ann_list = json.load(f)

        for item_idx, item in enumerate(ann_list):
            # each image is an item
            image_name = item['file_upload']
            image_id = len(images)
            width, height = None, None

            # skip tasks without annotations
            if not item['annotations']:
                logger.warning('No annotations found for item #' +
                               str(item_idx))
                continue

            kp_num = 0
            for i, label in enumerate(item['annotations'][0]['result']):
                category_name = None

                # valid label
                for key in [
                        'rectanglelabels', 'polygonlabels', 'labels',
                        'keypointlabels'
                ]:
                    if key == label['type'] and len(label['value'][key]) > 0:
                        category_name = label['value'][key][0]
                        break

                if category_name is None:
                    logger.warning('Unknown label type or labels are empty')
                    continue

                if not height or not width:
                    if 'original_width' not in label or \
                            'original_height' not in label:
                        logger.debug(
                            f'original_width or original_height not found'
                            f'in {image_name}')
                        continue

                    # get height and width info from annotations
                    width, height = label['original_width'], label[
                        'original_height']
                    images = add_image(images, width, height, image_id,
                                       image_name)

                category_id = self.category_name_to_id[category_name]

                annotation_id = len(annotations)

                if 'rectanglelabels' == label['type'] or 'labels' == label[
                        'type']:

                    x = label['value']['x']
                    y = label['value']['y']
                    w = label['value']['width']
                    h = label['value']['height']

                    x = x * label['original_width'] / 100
                    y = y * label['original_height'] / 100
                    w = w * label['original_width'] / 100
                    h = h * label['original_height'] / 100

                    # rect annotation should be later than keypoints
                    annotations[-1]['bbox'] = [x, y, w, h]
                    annotations[-1]['area'] = w * h
                    annotations[-1]['num_keypoints'] = kp_num

                elif 'polygonlabels' == label['type']:
                    points_abs = [(x / 100 * width, y / 100 * height)
                                  for x, y in label['value']['points']]
                    x, y = zip(*points_abs)

                    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)

                    # calculate bbox and area from polygon's points
                    # which may be different with rect annotation
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    area = float(0.5 * np.abs(
                        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

                    # polygon label should be later than keypoints
                    annotations[-1]['segmentation'] = [[
                        coord for point in points_abs for coord in point
                    ]]
                    annotations[-1]['bbox'] = bbox
                    annotations[-1]['area'] = area
                    annotations[-1]['num_keypoints'] = kp_num

                elif 'keypointlabels' == label['type']:
                    x = label['value']['x'] * label['original_width'] / 100
                    y = label['value']['y'] * label['original_height'] / 100

                    # there is no method to annotate visible in Label Studio
                    # so the keypoints' visible code will be 2 except (0,0)
                    if x == y == 0:
                        current_kp = [x, y, 0]
                        kp_num_change = 0
                    else:
                        current_kp = [x, y, 2]
                        kp_num_change = 1

                    # create new annotation in coco
                    # when the keypoint is the first point of an instance
                    if i == 0 or item['annotations'][0]['result'][
                            i - 1]['type'] != 'keypointlabels':
                        annotations.append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
                            'keypoints': current_kp,
                            'ignore': 0,
                            'iscrowd': 0,
                        })
                        kp_num = kp_num_change
                    else:
                        annotations[-1]['keypoints'].extend(current_kp)
                        kp_num += kp_num_change

        with io.open(output_json, mode='w', encoding='utf8') as fout:
            json.dump(
                {
                    'images': images,
                    'categories': self.categories,
                    'annotations': annotations,
                    'info': {
                        'year': datetime.now().year,
                        'version': '1.0',
                        'description': '',
                        'contributor': 'Label Studio',
                        'url': '',
                        'date_created': str(datetime.now()),
                    },
                },
                fout,
                indent=2,
            )


def main():
    args = parse_args()
    config = args.config
    input_json = args.input
    output_json = args.output
    converter = LSConverter(config)
    converter.convert_to_coco(input_json, output_json)


if __name__ == '__main__':
    main()
