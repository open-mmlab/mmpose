# Label Studio 标注工具转COCO脚本

[Label Studio](https://labelstud.io/) 是一款广受欢迎的深度学习标注工具，可以对多种任务进行标注，然而对于关键点标注，Label Studio 无法直接导出成 MMPose 所需要的 COCO 格式。本文将介绍如何使用Label Studio 标注关键点数据，并利用 [labelstudio2coco.py](../../../tools/dataset_converters/labelstudio2coco.py) 工具将其转换为训练所需的格式。

## Label Studio 标注要求

根据 COCO 格式的要求，每个标注的实例中都需要包含关键点、分割和 bbox 的信息，然而 Label Studio 在标注时会将这些信息分散在不同的实例中，因此需要按一定规则进行标注，才能正常使用后续的脚本。

1. 标签接口设置

对于一个新建的 Label Studio 项目，首先要设置它的标签接口。这里需要有三种类型的标注：`KeyPointLabels`、`PolygonLabels`、`RectangleLabels`，分别对应 COCO 格式中的`keypoints`、`segmentation`、`bbox`。以下是一个标签接口的示例，可以在项目的`Settings`中找到`Labeling Interface`，点击`Code`，粘贴使用该示例。

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

2. 标注顺序

由于需要将多个标注实例中的不同类型标注组合到一个实例中，因此采取了按特定顺序标注的方式，以此来判断各标注是否位于同一个实例。标注时须按照 **KeyPointLabels -> PolygonLabels/RectangleLabels** 的顺序标注，其中 KeyPointLabels 的顺序和数量要与 MMPose 配置文件中的`dataset_info`的关键点顺序和数量一致， PolygonLabels 和 RectangleLabels 的标注顺序可以互换，且可以只标注其中一个，只要保证一个实例的标注中，以关键点开始，以非关键点结束即可。下图为标注的示例：

*注：bbox 和 area 会根据靠后的 PolygonLabels/RectangleLabels 来计算，如若先标 PolygonLabels，那么bbox会是靠后的 RectangleLabels 的范围，面积为矩形的面积，反之则是多边形外接矩形和多边形的面积*

![image](https://github.com/open-mmlab/mmpose/assets/15847281/b2d004d0-8361-42c5-9180-cfbac0373a94)

3. 导出标注

上述标注完成后，需要将标注进行导出。选择项目界面的`Export`按钮，选择`JSON`格式，再点击`Export`即可下载包含标签的 JSON 格式文件。

*注：上述文件中仅仅包含标签，不包含原始图片，因此需要额外提供标注对应的图片。由于 Label Studio 会对过长的文件名进行截断，因此不建议直接使用上传的文件，而是使用`Export`功能中的导出 COCO 格式工具，使用压缩包内的图片文件夹。*

![image](https://github.com/open-mmlab/mmpose/assets/15847281/9f54ca3d-8cdd-4d7f-8ed6-494badcfeaf2)

## 转换工具脚本的使用

转换工具脚本位于`tools/dataset_converters/labelstudio2coco.py`，使用方式如下：

```bash
python tools/dataset_converters/labelstudio2coco.py config.xml project-1-at-2023-05-13-09-22-91b53efa.json output/result.json
```

其中`config.xml`的内容为标签接口设置中提到的`Labeling Interface`中的`Code`，`project-1-at-2023-05-13-09-22-91b53efa.json`即为导出标注时导出的 Label Studio 格式的 JSON 文件，`output/result.json`为转换后得到的 COCO 格式的 JSON 文件路径，若路径不存在，该脚本会自动创建路径。

随后，将图片的文件夹放置在输出目录下，即可完成 COCO 数据集的转换。目录结构示例如下：

```bash
.
├── images
│   ├── 38b480f2.jpg
│   └── aeb26f04.jpg
└── result.json

```

若想在 MMPose 中使用该数据集，可以进行类似如下的修改：

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
