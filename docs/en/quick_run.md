# Quick Run

This page provides a basic tutorial about the usage of MMPose.

We will walk you through the 7 key steps of a typical MMPose workflow by training a top-down residual log-likelihood algorithm based on resnet50 on COCO dataset:

1. Inference with a pretrained model
2. Prepare the dataset
3. Prepare a config
4. Browse the transformed images
5. Training
6. Testing
7. Visualization

## Installation

For installation instructions, please refer to [Installation](./installation.md).

## Get Started

### Inference with a pretrained model

We provide a useful script to perform pose estimation with a pretrained model:

```Bash
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_rle-2ea9bb4a_20220616.pth
```

If MMPose is properly installed, you will get the visualized result as follows:

![inference_demo](https://user-images.githubusercontent.com/13503330/187112344-0c5062f2-689c-445c-a259-d5d4311e2497.png)

```{note}
More demo and full instructions can be found in [Inference](./user_guides/inference.md).
```

### Prepare the dataset

MMPose supports multiple tasks. We provide the corresponding guidelines for data preparation.

- [2D Body Keypoint Detection](./dataset_zoo/2d_body_keypoint.md)

- [3D Body Keypoint Detection](./dataset_zoo/3d_body_keypoint.md)

- [2D Hand Keypoint Detection](./dataset_zoo/2d_hand_keypoint.md)

- [3D Hand Keypoint Detection](./dataset_zoo/3d_hand_keypoint.md)

- [2D Face Keypoint Detection](./dataset_zoo/2d_face_keypoint.md)

- [2D WholeBody Keypoint Detection](./dataset_zoo/2d_wholebody_keypoint.md)

- [2D Fashion Landmark Detection](./dataset_zoo/2d_fashion_landmark.md)

- [2D Animal Keypoint Detection](./dataset_zoo/2d_animal_keypoint.md)

You can refer to \[2D Body Keypoint Detection\] > \[COCO\] for COCO dataset preparation.

```{note}
In MMPose, we suggest placing the data under `$MMPOSE/data`.
```

### Prepare a config

MMPose is equipped with a powerful config system to conduct various experiments conveniently. A config file organizes the settings of:

- **General**: basic configurations non-related to training or testing, such as Timer, Logger, Visualizer and other Hooks, as well as distributed-related environment settings

- **Data**: dataset, dataloader and data augmentation

- **Training**: resume, weights loading, optimizer, learning rate scheduling, epochs and valid interval etc.

- **Model**: structure, module and loss function etc.

- **Evaluation**: metrics

We provide a bunch of well-prepared configs under `$MMPOSE/configs` so that you can directly use or modify.

Going back to our example, we  will use the prepared config:

```Bash
$MMPOSE/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py
```

You can set the path of the COCO dataset by modifying `data_root` in the config：

```Python
data_root = 'data/coco'
```

```{note}
If you wish to learn more about our config system, please refer to [Configs](./user_guides/configs.md).
```

### Browse the transformed images

Before training, we can browse the transformed training data to check if the images are augmented properly:

```Bash
python tools/misc/browse_dastaset.py \
    configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py \
    --mode transformed
```

![transformed_training_img](https://user-images.githubusercontent.com/13503330/187112376-e604edcb-46cc-4995-807b-e8f204f991b0.png)

### Training

Use the following command to train with a single GPU:

```Bash
python tools/train.py configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py
```

```{note}
MMPose automates many useful training tricks and functions including:

- Learning rate warmup and scheduling

- ImageNet pretrained models

- Automatic learning rate scaling

- Multi-GPU and Multi-Node training support

- Various Data backend support, e.g. HardDisk, LMDB, Petrel, HTTP etc.

- Mixed precision training support

- TensorBoard
```

### Testing

Checkpoints and logs will be saved under `$MMPOSE/work_dirs` by default. The best model is under `$MMPOSE/work_dir/best_coco`.

Use the following command to evaluate the model on COCO dataset:

```Bash
python tools/test.py \
    configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_rle-8xb64-210e_coco-256x192.py \
    work_dir/best_coco/AP_epoch_20.pth
```

Here is an example of evaluation results：

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
If you want to perform evaluation on other datasets, please refer to [Train & Test](./user_guides/train_and_test.md).
```

### Visualization

In addition to the visualization of the keypoint skeleton, MMPose also supports the visualization of Heatmaps by setting `output_heatmap=True` in confg:

```Python
model = dict(
    ## omitted
    test_cfg = dict(
        ## omitted
        output_heatmaps=True
    )
)
```

or add `--cfg-options='model.test_cfg.output_heatmaps=True'` at the end of your command.

Visualization result (top: decoded keypoints; bottom: predicted heatmap):

![vis_pred](https://user-images.githubusercontent.com/26127467/187578902-30ef7bb0-9a93-4e03-bae0-02aeccf7f689.jpg)

```{note}
If you wish to apply MMPose to your own projects, we have prepared a detailed [Migration guide](./migration.md).
```
