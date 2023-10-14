# Label Studio Annotations to COCO Script

[Label Studio](https://labelstud.io/) is a popular deep learning annotation tool that can be used for annotating various tasks. However, for keypoint annotation, Label Studio can not directly export to the COCO format required by MMPose. This article will explain how to use Label Studio to annotate keypoint data and convert it into the required COCO format using the [labelstudio2coco.py](../../../tools/dataset_converters/labelstudio2coco.py) tool.

## Label Studio Annotation Requirements

According to the COCO format requirements, each annotated instance needs to include information about keypoints, segmentation, and bounding box (bbox). However, Label Studio scatters this information across different instances during annotation. Therefore, certain rules need to be followed during annotation to ensure proper usage with the subsequent scripts.

1. Label Interface Setup

For a newly created Label Studio project, the label interface needs to be set up. There should be three types of annotations: `KeyPointLabels`, `PolygonLabels`, and `RectangleLabels`, which correspond to `keypoints`, `segmentation`, and `bbox` in the COCO format, respectively. The following is an example of a label interface. You can find the `Labeling Interface` in the project's `Settings`, click on `Code`, and paste the following example.

```xml
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

2. Annotation Order

Since it is necessary to combine annotations of different types into one instance, a specific order of annotation is required to determine whether the annotations belong to the same instance. Annotations should be made in the order of `KeyPointLabels` -> `PolygonLabels`/`RectangleLabels`. The order and number of `KeyPointLabels` should match the order and number of keypoints specified in the `dataset_info` in MMPose configuration file. The annotation order of `PolygonLabels` and `RectangleLabels` can be interchangeable, and only one of them needs to be annotated. The annotation should be within one instance starts with keypoints and ends with non-keypoints. The following image shows an annotation example:

*Note: The bbox and area will be calculated based on the later PolygonLabels/RectangleLabels. If you annotate PolygonLabels first, the bbox will be based on the range of the later RectangleLabels, and the area will be equal to the area of the rectangle. Conversely, they will be based on the minimum bounding rectangle of the polygon and the area of the polygon.*

![image](https://github.com/open-mmlab/mmpose/assets/15847281/b2d004d0-8361-42c5-9180-cfbac0373a94)

3. Exporting Annotations

Once the annotations are completed as described above, they need to be exported. Select the `Export` button on the project interface, choose the `JSON` format, and click `Export` to download the JSON file containing the labels.

*Note: The exported file only contains the labels and does not include the original images. Therefore, the corresponding annotated images need to be provided separately. It is not recommended to use directly uploaded files because Label Studio truncates long filenames. Instead, use the export COCO format tool available in the `Export` functionality, which includes a folder with the image files within the downloaded compressed package.*

![image](https://github.com/open-mmlab/mmpose/assets/15847281/9f54ca3d-8cdd-4d7f-8ed6-494badcfeaf2)

## Usage of the Conversion Tool Script

The conversion tool script is located at `tools/dataset_converters/labelstudio2coco.py`and can be used as follows:

```bash
python tools/dataset_converters/labelstudio2coco.py config.xml project-1-at-2023-05-13-09-22-91b53efa.json output/result.json
```

Where `config.xml` contains the code from the Labeling Interface mentioned earlier, `project-1-at-2023-05-13-09-22-91b53efa.json` is the JSON file exported from Label Studio, and `output/result.json` is the path to the resulting JSON file in COCO format. If the path does not exist, the script will create it automatically.

Afterward, place the image folder in the output directory to complete the conversion of the COCO dataset. The directory structure can be as follows:

```bash
.
├── images
│   ├── 38b480f2.jpg
│   └── aeb26f04.jpg
└── result.json

```

If you want to use this dataset in MMPose, you can make modifications like the following example:

```python
dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='result.json',
    data_prefix=dict(img='images/'),
    pipeline=train_pipeline,
)
```
