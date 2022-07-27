<!-- TOC -->

- [常见问题](#常见问题)
  - [安装](#安装)
  - [开发](#开发)
  - [数据](#数据)
  - [训练](#训练)
  - [评测](#评测)
  - [推理](#推理)
  - [部署](#部署)

<!-- TOC -->

# 常见问题

我们在这里列出了一些常见问题及其相应的解决方案。
如果您发现任何常见问题并有方法帮助解决，欢迎随时丰富列表。
如果这里的内容没有涵盖您的问题，请按照[提问模板](https://github.com/open-mmlab/mmpose/issues/new/choose)在 GitHub 上提出问题，并补充模板中需要的信息。

## 安装

- MMCV 与 MMPose 的兼容问题。如 "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

  这里列举了各版本 MMPose 对 MMCV 版本的依赖，请选择合适的 MMCV 版本来避免安装和使用中的问题。

| MMPose 版本 |         MMCV 版本         |
| :---------: | :-----------------------: |
|   master    | mmcv-full>=1.3.8, \<1.7.0 |
|   0.26.0    | mmcv-full>=1.3.8, \<1.6.0 |
|   0.25.1    | mmcv-full>=1.3.8, \<1.6.0 |
|   0.25.0    | mmcv-full>=1.3.8, \<1.5.0 |
|   0.24.0    | mmcv-full>=1.3.8, \<1.5.0 |
|   0.23.0    | mmcv-full>=1.3.8, \<1.5.0 |
|   0.22.0    | mmcv-full>=1.3.8, \<1.5.0 |
|   0.21.0    | mmcv-full>=1.3.8, \<1.5.0 |
|   0.20.0    | mmcv-full>=1.3.8, \<1.4.0 |
|   0.19.0    | mmcv-full>=1.3.8, \<1.4.0 |
|   0.18.0    | mmcv-full>=1.3.8, \<1.4.0 |
|   0.17.0    | mmcv-full>=1.3.8, \<1.4.0 |
|   0.16.0    | mmcv-full>=1.3.8, \<1.4.0 |
|   0.14.0    | mmcv-full>=1.1.3, \<1.4.0 |
|   0.13.0    | mmcv-full>=1.1.3, \<1.4.0 |
|   0.12.0    |  mmcv-full>=1.1.3, \<1.3  |
|   0.11.0    |  mmcv-full>=1.1.3, \<1.3  |
|   0.10.0    |  mmcv-full>=1.1.3, \<1.3  |
|    0.9.0    |  mmcv-full>=1.1.3, \<1.3  |
|    0.8.0    |  mmcv-full>=1.1.1, \<1.2  |
|    0.7.0    |  mmcv-full>=1.1.1, \<1.2  |

- **无法安装 xtcocotools**

  1. 尝试使用 pip 手动安装：`pip install xtcocotools`.
  2. 如果第一步无法安装，尝试从[源码](https://github.com/jin-s13/xtcocoapi)安装：

  ```
  git clone https://github.com/jin-s13/xtcocoapi
  cd xtcocoapi
  python setup.py install
  ```

- **报错: No matching distribution found for xtcocotools>=1.6**

  1. 安装 cython : `pip install cython`.
  2. 从[源码](https://github.com/jin-s13/xtcocoapi) 安装 xtcocotools ：

  ```
  git clone https://github.com/jin-s13/xtcocoapi
  cd xtcocoapi
  python setup.py install
  ```

- **报错："No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'"**

  1. 如果您已经安装了 `mmcv`, 您需要运行 `pip uninstall mmcv` 来卸载已经安装的 `mmcv` 。如果您在本地同时安装了 `mmcv` 和 `mmcv-full`, `ModuleNotFoundError` 将会抛出。
  2. 按照[安装指南](https://mmcv.readthedocs.io/zh_CN/latest/#installation)安装 `mmcv-full`.

## 开发

- 如果对源码进行了改动，需要重新安装以使改动生效吗？

  如果您遵照[最佳实践](install.md#最佳实践)的指引，从源码安装 `mmpose`，那么任何本地修改都不需要重新安装即可生效。

- 如何在多个 `MMPose` 版本下进行开发？

  通常来说，我们推荐通过不同虚拟环境来管理多个开发目录下的 `MMPose`. 但如果您希望在不同目录（如 `mmpose-0.26.0`, `mmpose-0.25.0` 等）使用同一个环境进行开发，我们提供的训练和测试 shell 脚本会自动使用当前目录的 `mmpose`，其他 Python 脚本则可以在命令前添加 `` PYTHONPATH=`pwd`  `` 来使用当前目录的代码。

  反过来，如果您希望 shell 脚本使用环境中安装的 `MMPose`，而不是当前目录的，则可以去掉 shell 脚本中如下一行代码：

  ```shell
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
  ```

  ## 数据

  - **怎么将 2D 关键点数据集转换为 COCO 格式？**

  您可以参考这个[转换工具](https://github.com/open-mmlab/mmpose/blob/master/tools/dataset/parse_macaquepose_dataset.py) 来准备您的数据。
  这是一个关于 COCO 格式 json 文件的[示例](https://github.com/open-mmlab/mmpose/blob/master/tests/data/macaque/test_macaque.json)。
  COCO 格式的 json 文件需要这些字段信息: "`categories`", "`annotations`" and "`images`".
  "`categories`" 包括了数据集的一些基本信息，如类别名称和关键点名称。"`images`" 包含了图片级别的信息，需要这些字段的信息："`id`", "`file_name`", "`height`", "`width`". 其他字段是可选的。
  注： "`id`" 可以是不连续或者没有排序好的（如 1000, 40, 352, 333 ...）。

  "`annotations`" 包含了实例级别的信息，需要这些字段的信息："`image_id`", "`id`", "`keypoints`", "`num_keypoints`", "`bbox`", "`iscrowd`", "`area`", "`category_id`". 其他字段是可选的。
  注：(1) "`num_keypoints`" 表示可见关键点的数量. (2) 默认情况下，请设置"`iscrowd: 0`". (3) "`area`" 可以通过标记的边界框信息计算得到：(area = w * h). (4) 简单地设置 "`category_id: 1`" 即可. (5) "`annotations`" 中的 "`image_id`" 应该和 "`images`" 中的 "`id`" 匹配。

  - **如果数据集没有人工标注的边界框信息怎么办？**

  我们可以认为一个人的边界框是恰好包围所有关键点的最小框。

  - **如果数据集没有 `segmentation` 信息怎么办？**

  设置人体的 `area` 为边界框的面积即可。在评测的时候，请按照这个[例子](https://github.com/open-mmlab/mmpose/blob/a82dd486853a8a471522ac06b8b9356db61f8547/mmpose/datasets/datasets/top_down/topdown_aic_dataset.py#L113) 设置 `use_area=False`。

  - **`COCO_val2017_detections_AP_H_56_person.json` 是什么文件？可以不使用它来训练姿态估计的模型吗？**

  `COCO_val2017_detections_AP_H_56_person.json` 包含了在 COCO 验证集上**检测**到的人体边界框，是使用 FasterRCNN 生成的。
  您可以使用真实标记的边界框来评测模型，设置 `use_gt_bbox=True` 和 `bbox_file=''` 即可。
  或者您可以使用**检测**到的边界框来评测模型的泛化性，只要设置 `use_gt_bbox=False` 和 `bbox_file='COCO_val2017_detections_AP_H_56_person.json'` 即可。

  ## 训练

  - **报错：RuntimeError: Address already in use**

  设置环境变量 `MASTER_PORT=XXX`。
  例如 `MASTER_PORT=29517 GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=2 ./tools/slurm_train.sh Test res50 configs/body/2D_Kpt_SV_RGB_Img/topdown_hm/coco/res50_coco_256x192.py work_dirs/res50_coco_256x192`。

- **加载模型过程中："Unexpected keys in source state dict" when loading pre-trained weights**

  在姿态估计模型中不使用预训练模型中的某些层是正常的。在 ImageNet 预训练的分类网络和姿态估计网络可能具有不同的架构（例如，没有分类头）。因此，在源模型权重文件 (source state dict) 中确实会出现一些预期之外的键。

- **怎么使用经过训练的模型作为主干网络的预训练**

  如果要对整个网络（主干网络 + 头部网络）使用预训练模型，
  请参考教程文档：[使用预训练模型](/docs/zh_CN/tutorials/1_finetune.md#使用预训练模型)，配置文件中的 `load_from` 字段指明了预训练模型的链接。

  如果要使用主干网进行预训练，可以将配置文件中主干网络的 "`pretrained`" 值改为模型权重文件的路径或者 URL 。
  训练时，将忽略意外的键值。

- **怎么实时地可视化训练的准确率/损失函数曲线？**

  在 `log_config` 中使用 `TensorboardLoggerHook`，如：

  ```python
  log_config=dict(interval=20, hooks=[dict(type='TensorboardLoggerHook')])
  ```

  您还可以参考教程文档:[自定义运行配置](tutorials/6_customize_runtime.md#日志配置) 以及配置文件的[例子](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L26)。

- **没有打印日志信息**

  使用更小的日志打印间隔。例如，将这个[配置文件](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L23)的 `interval=50` 改为`interval=1`.

- **微调模型的时候怎么固定主干网络多个阶段的网络参数？**

  您可以参考这个函数: [`def _freeze_stages()`](https://github.com/open-mmlab/mmpose/blob/d026725554f9dc08e8708bd9da8678f794a7c9a6/mmpose/models/backbones/resnet.py#L618) 以及这个参数：[`frozen_stages`](https://github.com/open-mmlab/mmpose/blob/d026725554f9dc08e8708bd9da8678f794a7c9a6/mmpose/models/backbones/resnet.py#L498)。
  如果使用分布式训练或者测试，请在配置文件中设置 `find_unused_parameters = True`。

## 评测

- **怎么在 MPII 测试集上运行评测？**

  因为我们没有 MPII 测试集上的真实标注信息，我们不能在**本地**评测。
  如果您获得在测试集上的评测结果，根据 [MPII 指南](http://human-pose.mpi-inf.mpg.de/#evaluation)，您需要通过邮件上传这个文件 `pred.mat` （在测试过程中生成）到官方的服务器。

- **对于自顶向下的 2D 姿态估计方法，为什么预测的关键点坐标可以超出边界框？**

  `MMPose` 没有直接使用边界框来裁剪图片。边界框首先会被转换至中心和尺度，尺度会乘上一个系数 (1.25) 来包含一些图片的上下文信息。如果图片的宽高比和模型的输入 (通常是 192/256) 不一致，会调整这个边界框的大小。详细可参考[代码](https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/pipelines/top_down_transform.py#L15)。

## 推理

- **怎么在 CPU 上运行 MMPose ？**

  运行示例的时候设置: `--device=cpu`.

- **怎么加快推理速度？**

  对于自顶向下的模型，可以尝试修改配置文件。例如：

  1. 在 [topdown-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L51) 中设置 `flip_test=False`。
  2. 在 [topdown-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L52) 中设置 `post_process='default'`。
  3. 使用更快的人体边界框检测器，可参考 [MMDetection](https://mmdetection.readthedocs.io/zh_CN/latest/model_zoo.html)。

  对于自底向上的模型，也可以尝试修改配置文件。例如：

  1. 在 [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L91) 中设置 `flip_test=False`。
  2. 在 [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L89) 中设置 `adjust=False`。
  3. 在 [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L90) 中设置 `refine=False`。
  4. 在 [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L39) 中使用更小的输入图片尺寸。

## 部署

- **为什么用 MMPose 转换导出的 onnx 模型在转移到其他框架如 TensorRT 时会抛出错误？**

  目前，我们只能确保 `MMPose` 中的模型与 onnx 兼容。但是您的目标部署框架可能不支持 onnx 中的某些操作，例如这个[问题](https://github.com/open-mmlab/mmaction2/issues/414) 中提到的 TensorRT 。

  请注意，`MMPose` 中的 `pytorch2onnx` 将不再维护，未来将不再保留。我们将在 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 中支持所有 `OpenMMLab` 代码库的模型部署，包括 `MMPose`。您可以在其[文档](https://mmdeploy.readthedocs.io/en/latest/)中找到有关受支持模型和用户指南的详细信息，并提出问题以请求支持您要使用的模型。
