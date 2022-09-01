# 快速上手

在这一章里，我们将带领你走过MMPose工作流程中关键的七个步骤，帮助你快速上手：

1. 使用预训练模型进行推理
2. 准备数据集
3. 准备配置文件
4. 可视化训练图片
5. 训练
6. 测试
7. 可视化

## 安装

请查看[安装指南](./installation.md)，以了解完整步骤。

## 快速开始

### 使用预训练模型进行推理

你可以通过以下命令来使用预训练模型对单张图片进行识别：

```Bash
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py\
    https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_rle-2ea9bb4a_20220616.pth
```

该命令中用到了测试图片、完整的配置文件、预训练模型，如果MMPose安装无误，将会弹出一个新窗口，对检测结果进行可视化显示：

![inference_demo](https://user-images.githubusercontent.com/13503330/187112344-0c5062f2-689c-445c-a259-d5d4311e2497.png)

更多演示脚本的详细参数说明可以在 [模型推理](./user_guides/inference.md) 中找到。

### 准备数据集

MMPose支持各种不同的任务，我们提供了对应的数据集准备教程。

- [2D人体关键点](./dataset_zoo/2d_body_keypoint.md)

- [3D人体关键点](./dataset_zoo/3d_body_keypoint.md)

- [2D人手关键点](./dataset_zoo/2d_hand_keypoint.md)

- [3D人手关键点](./dataset_zoo/3d_hand_keypoint.md)

- [2D人脸关键点](./dataset_zoo/2d_face_keypoint.md)

- [2D全身人体关键点](./dataset_zoo/2d_wholebody_keypoint.md)

- [2D服饰关键点](./dataset_zoo/2d_fashion_landmark.md)

- [2D动物关键点](./dataset_zoo/2d_animal_keypoint.md)

你可以在【2D人体关键点数据集】>【COCO】下找到COCO数据集的准备教程，并按照教程完成数据集的下载和整理。

```{note}
在MMPose中，我们建议将COCO数据集存放到新建的 `$MMPOSE/data` 目录下。
```

### 准备配置文件

MMPose拥有一套强大的配置系统，用于管理训练所需的一系列必要参数：

- **通用**：环境、Hook、Checkpoint、Logger、Timer等

- **数据**：Dataset、Dataloader、数据增强等

- **训练**：优化器、学习率调整等

- **模型**：Backbone、Neck、Head、损失函数等

- **评测**：Metrics

在`$MMPOSE/configs`目录下，我们提供了大量前沿论文方法的配置文件，可供直接使用和参考。

要在COCO数据集上训练基于ResNet50的RLE模型时，所需的配置文件为：

```Bash
$MMPOSE/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py
```

我们需要将配置文件中的 data_root 变量修改为COCO数据集存放路径：

```Python
data_root = 'data/coco'
```

```{note}
感兴趣的读者也可以查阅 [配置文件](./user_guides/configs.md) 来进一步学习MMPose所使用的配置系统。
```

### 可视化训练图片

在开始训练之前，我们还可以对训练图片进行可视化，检查训练图片是否正确进行了数据增强。

我们提供了相应的可视化脚本：

```Bash
python tools/misc/browse_dastaset.py \
    configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py \
    --mode transformed
```

![transformed_training_img](https://user-images.githubusercontent.com/13503330/187112376-e604edcb-46cc-4995-807b-e8f204f991b0.png)

### 训练

确定数据无误后，运行以下命令启动训练：

```Bash
python tools/train.py configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py
```

```{note}
MMPose中集成了大量实用训练trick和功能：

- 学习率warmup和scheduling

- ImageNet预训练权重

- 自动学习率缩放、自动batch size缩放

- CPU训练、多机多卡训练、集群训练

- HardDisk、LMDB、Petrel、HTTP等不同数据后端

- 混合精度浮点训练

- TensorBoard
```

### 测试

在不指定额外参数时，训练的权重和日志信息会默认存储到`$MMPOSE/work_dirs`目录下，最优的模型权重存放在`$MMPOSE/work_dir/best_coco`目录下。

我们可以通过如下指令测试模型在COCO验证集上的精度：

```Bash
python tools/test.py \
    configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py \
    work_dir/best_coco/AP_epoch_20.pth
```

在COCO验证集上评测结果样例如下：

```Bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.704
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.883
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.777
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.667
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.751
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.920
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.815
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.709
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.811
08/23 12:04:42 - mmengine - INFO - Epoch(test) [3254/3254]  coco/AP: 0.704168  coco/AP .5: 0.883134  coco/AP .75: 0.777015  coco/AP (M): 0.667207  coco/AP (L): 0.768644  coco/AR: 0.750913  coco/AR .5: 0.919710  coco/AR .75: 0.815334  coco/AR (M): 0.709232  coco/AR (L): 0.811334
```

```{note}
如果需要测试模型在其他数据集上的表现，可以前往 [训练与测试](./user_guides/train_and_test.md) 查看。
```

### 可视化

除了对关键点骨架的可视化以外，我们还支持对热度图进行可视化，你只需要在配置文件中设置`output_heatmap=True`：

```Python
model = dict(
    ## 内容省略
    test_cfg = dict(
        ## 内容省略
        output_heatmaps=True
    )
)
```

或在命令行中添加`--cfg-options='model.test_cfg.output_heatmaps=True'`。

可视化效果如下：

![vis_pred](https://user-images.githubusercontent.com/26127467/187578902-30ef7bb0-9a93-4e03-bae0-02aeccf7f689.jpg)

```{note}
如果你希望深入地学习MMPose，将其应用到自己的项目当中，我们准备了一份详细的 [迁移指南](./migration.md) 。
```
