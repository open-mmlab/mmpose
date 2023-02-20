# Inference with existing models

MMPose provides hundreds of pre-trained models for pose estimation in [Model Zoo](https://mmpose.readthedocs.io/en/1.x/modelzoo.html).
This note will show **how to perform inference**, which means running pose estimation on given images or videos with trained models.

As for how to test existing models on standard datasets, please see this [guide](./train_and_test.md#test).

In MMPose, a model is defined by a configuration file and existing model parameters are saved in a checkpoint file.

To start with, we recommend HRNet model with [this configuration file](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) and [this checkpoint file](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth). It is recommended to download the checkpoint file to `checkpoints` directory.

## High-level APIs for inference

MMPose offers a comprehensive high-level API for inference, known as `MMPoseInferencer`. This API enables users to perform inference on both images and videos using all the models supported by MMPose. Furthermore, the API provides automatic visualization of inference results and allows for the convenient saving of predictions.

Here is an example of inference on a given image using the pre-trained human pose estimator.

```python
from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/000000000785.jpg'   # you can specify your own picture path

# build the inferencer with model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API utilizes a lazy inference strategy,
# whereby it generates a prediction generator when provided with input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
```

If everything works fine, you will see the following image in a new window.
![inferencer_result_coco](https://user-images.githubusercontent.com/26127467/220008302-4a57fd44-0978-408e-8351-600e5513316a.jpg)

The variable `result` is a dictionary that contains two keys, `'visualization'` and `'predictions'`. The key `'visualization'` is intended to contain the visualization results. However, as the `return_vis` argument was not specified, this list remains blank. On the other hand, the key `'predictions'` is a list that contains the estimated keypoints for each individual instance.

### CLI tool

A command-line interface (CLI) tool for the inferencer is also available: `demo/inferencer_demo.py`. This tool enables users to perform inference with the same model and inputs using the following command:

```bash
python demo/inferencer_demo.py 'tests/data/coco/000000000785.jpg' \
    --pose2d 'human' --show --pred-out-dir 'predictions'
```

The predictions will be save in `predictions/000000000785.json`.

### Custom pose estimation models

The inferencer provides several methods that can be used to customize the models employed:

```python

# build the inferencer with model alias
# the available aliases include 'human', 'hand', 'face' and 'animal'
inferencer = MMPoseInferencer('human')

# build the inferencer with model config name
inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')

# build the inferencer with model config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/' \
           'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    pose2d_weights='https://download.openmmlab.com/mmpose/top_down/' \
                   'hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
)
```

In addition, top-down pose estimators also require an object detection model. The inferencer is capable of inferring the instance type for models trained with datasets supported in MMPose, and subsequently constructing the necessary object detection model. Alternatively, users may also manually specify the detection model using the following methods:

```python

# specify detection model by alias
# the available aliases include 'human', 'hand', 'face', 'animal',
# as well as any additional aliases defined in mmdet
inferencer = MMPoseInferencer(
    # suppose the pose estimator is trained on custom dataset
    pose2d='custom_human_pose_estimator.py',
    pose2d_weights='custom_human_pose_estimator.pth',
    det_model='human'
)

# specify detection model with model config name
inferencer = MMPoseInferencer(
    pose2d='human',
    det_model='yolox_l_8x8_300e_coco',
    det_cat_ids=[0],  # the category id of 'human' class
)

# specify detection model with config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose2d='human',
    det_model=f'{PATH_TO_MMDET}/configs/yolox/yolox_l_8x8_300e_coco.py',
    det_weights='https://download.openmmlab.com/mmdetection/v2.0/' \
                'yolox/yolox_l_8x8_300e_coco/' \
                'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    det_cat_ids=[0],  # the category id of 'human' class
)
```

### Input format

The inferencer is capable of processing a range of input types, which includes the following:

- A path to an image
- A path to a video
- A path to a folder (which will cause all images in that folder to be inferred)
- An image array
- A list of image arrays
- A webcam (in which case the `input` parameter should be set to either `'webcam'` or `'webcam:{CAMERA_ID}'`)

### Output settings

The inferencer is capable of both visualizing and saving predictions. The relevant arguments are as follows:

| Argument            | Description                                                                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `show`              | Determines whether the image or video should be displayed in a pop-up window.                                                                      |
| `radius`            | Sets the keypoint radius for visualization.                                                                                                        |
| `thickness`         | Sets the link thickness for visualization.                                                                                                         |
| `return_vis`        | Determines whether visualization images should be included in the results.                                                                         |
| `vis_out_dir`       | Specifies the folder path for saving the visualization images. If not set, the visualization images will not be saved.                             |
| `return_datasample` | Determines whether to return the prediction in the format of `PoseDataSample`.                                                                     |
| `pred_out_dir`      | Specifies the folder path for saving the predictions. If not set, the predictions will not be saved.                                               |
| `out_dir`           | If `vis_out_dir` or `pred_out_dir` is not set, the values will be set to `f'{out_dir}/visualization'` or `f'{out_dir}/predictions'`, respectively. |

## Demos

We also provide demo scripts, implemented with high-level APIs and supporting various tasks. Source codes are available [here](/demo). You can refer to the [docs](/demo/docs) for detail descriptions
