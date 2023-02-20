# Inference with existing models

MMPose provides hundreds of pre-trained models for pose estimation in [Model Zoo](https://mmpose.readthedocs.io/en/1.x/modelzoo.html).
This note will show **how to perform inference**, which means running pose estimation on given images or videos with trained models.

As for how to test existing models on standard datasets, please see this [guide](./train_and_test.md#test).

In MMPose, a model is defined by a configuration file and existing model parameters are saved in a checkpoint file.

To start with, we recommend HRNet model with [this configuration file](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) and [this checkpoint file](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth). It is recommended to download the checkpoint file to `checkpoints` directory.

## Out-of-the-box inferencer

MMPose offers a comprehensive API for inference, known as `MMPoseInferencer`. This API enables users to perform inference on both images and videos using all the models supported by MMPose. Furthermore, the API provides automatic visualization of inference results and allows for the convenient saving of predictions.

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

## High-level APIs for inference

MMPose provides high-level Python APIs for inference on a given image:

- [init_model](/mmpose/apis/inference.py#L64): Initialize a model with a config and checkpoint
- [inference_topdown](/mmpose/apis/inference.py#L124): Conduct inference with the top-down pose estimator on a given image

Here is an example of building the model and inference on a given image using the pre-trained checkpoint of HRNet model on COCO dataset.

```python
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

config_path = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
checkpoint_path = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth' # can be a local path
img_path = 'tests/data/coco/000000000785.jpg'   # you can specify your own picture path

# register all modules and set mmpose as the default scope.
register_all_modules()
# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
# test a single image
result = inference_topdown(model, img_path)[0]

```

`result` is a `PoseDataSample` containing `gt_instances` and `pred_instances`. And `pred_instances` contains the prediction results, usually containing `keypoints`, `keypoint_scores`. The content of `result.pred_instances` is as follows:

```python
<InstanceData(

    META INFORMATION

    DATA FIELDS
    keypoints: array([[[365.83333333,  83.33333826],
                [365.83333333,  75.00000525],
                [365.83333333,  75.00000525],
                [382.5       ,  83.33333826],
                [365.83333333,  83.33333826],
                [399.16666667, 116.66667032],
                [365.83333333, 125.00000334],
                [440.83333333, 158.3333354 ],
                [340.83333333, 158.3333354 ],
                [449.16666667, 166.66666842],
                [299.16666667, 175.00000143],
                [432.5       , 208.33333349],
                [415.83333333, 216.66666651],
                [432.5       , 283.33333063],
                [374.16666667, 274.99999762],
                [482.5       , 366.66666079],
                [407.5       , 341.66666174]]])
    bbox_scores: array([1.], dtype=float32)
    bboxes: array([[  0.,   0., 640., 425.]], dtype=float32)
    keypoint_scores: array([[0.9001359 , 0.90607893, 0.8974595 , 0.8780644 , 0.8363602 ,
                0.86385334, 0.86548805, 0.53965414, 0.8379145 , 0.77825487,
                0.9050092 , 0.8631748 , 0.8176921 , 0.9184168 , 0.9040103 ,
                0.7687361 , 0.9573005 ]], dtype=float32)
) at 0x7f5785582df0>
```

An image demo can be found in [demo/image_demo.py](/demo/image_demo.py).

## Demos

We also provide demo scripts, implemented with high-level APIs and supporting various tasks. Source codes are available [here](/demo). You can refer to the [docs](/demo/docs) for detail descriptions
