# Inference with existing models

MMPose provides a wide variety of pre-trained models for pose estimation, which can be found in the [Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html).
This guide will demonstrate **how to perform inference**, or running pose estimation on provided images or videos using trained models.

For instructions on testing existing models on standard datasets, refer to this [guide](./train_and_test.md#test).

In MMPose, we provide two ways to perform inference:

1. Inferencer: a Unified Inference Interface
2. Python API: more flexible and customizable

## Inferencer: a Unified Inference Interface

MMPose offers a comprehensive API for inference, known as [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24). This API enables users to perform inference on both images and videos using all the models supported by MMPose. Furthermore, the API provides automatic visualization of inference results and allows for the convenient saving of predictions.

### Basic Usage

The [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) can be used in any Python program to perform pose estimation. Below is an example of inference on a given image using the pre-trained human pose estimator within the Python shell.

```python
from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/000000000785.jpg'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
```

If everything works fine, you will see the following image in a new window:
![inferencer_result_coco](https://user-images.githubusercontent.com/26127467/220008302-4a57fd44-0978-408e-8351-600e5513316a.jpg)

The `result` variable is a dictionary comprising two keys, `'visualization'` and `'predictions'`.

- `'visualization'` holds a list which:

  - contains visualization results, such as the input image, markers of the estimated poses, and optional predicted heatmaps.
  - remains empty if the `return_vis` argument is not specified.

- `'predictions'` stores:

  - a list of estimated keypoints for each identified instance.

The structure of the `result` dictionary is as follows:

```python
result = {
    'visualization': [
        # number of elements: batch_size (defaults to 1)
        vis_image_1,
        ...
    ],
    'predictions': [
        # pose estimation result of each image
        # number of elements: batch_size (defaults to 1)
        [
            # pose information of each detected instance
            # number of elements: number of detected instances
            {'keypoints': ...,  # instance 1
            'keypoint_scores': ...,
            ...
            },
            {'keypoints': ...,  # instance 2
            'keypoint_scores': ...,
            ...
            },
        ]
    ...
    ]
}

```

A **command-line interface (CLI)** tool for the inferencer is also available: `demo/inferencer_demo.py`. This tool allows users to perform inference using the same model and inputs with the following command:

```bash
python demo/inferencer_demo.py 'tests/data/coco/000000000785.jpg' \
    --pose2d 'human' --show --pred-out-dir 'predictions'
```

The predictions will be save in `predictions/000000000785.json`. The argument names correspond with the [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24), which serves as an API.

The inferencer is capable of processing a range of input types, which includes the following:

- A path to an image
- A path to a video
- A path to a folder (which will cause all images in that folder to be inferred)
- An image array (NA for CLI tool)
- A list of image arrays (NA for CLI tool)
- A webcam (in which case the `input` parameter should be set to either `'webcam'` or `'webcam:{CAMERA_ID}'`)

Please note that when the input corresponds to multiple images, such as when the input is a video or a folder path, the inference process needs to iterate over the results generator in order to perform inference on all the frames or images within the folder. Here's an example in Python:

```python
folder_path = 'tests/data/coco'

result_generator = inferencer(folder_path, show=True)
results = [result for result in result_generator]
```

In this example, the `inferencer` takes the `folder_path` as input and returns a generator object (`result_generator`) that produces inference results. By iterating over the `result_generator` and storing each result in the `results` list, you can obtain the inference results for all the frames or images within the folder.

### Custom Pose Estimation Models

The inferencer provides several methods that can be used to customize the models employed:

```python

# build the inferencer with model alias
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

The complere list of model alias can be found in the [Model Alias](#model-alias) section.

**Custom Inferencer for 3D Pose Estimation Models**

The code shown above provides examples for creating 2D pose estimator inferencers. You can similarly construct a 3D model inferencer by using the `pose3d` argument:

```python
# build the inferencer with 3d model alias
inferencer = MMPoseInferencer(pose3d="human3d")

# build the inferencer with 3d model config name
inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")

# build the inferencer with 3d model config path and checkpoint path/URL
inferencer = MMPoseInferencer(
    pose3d='configs/body_3d_keypoint/motionbert/h36m/' \
           'motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py',
    pose3d_weights='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/' \
                   'pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth'
)
```

**Custom Object Detector for Top-down Pose Estimation Models**

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

To perform top-down pose estimation on cropped images containing a single object, users can set `det_model='whole_image'`. This bypasses the object detector initialization, creating a bounding box that matches the input image size and directly sending the entire image to the top-down pose estimator.

### Dump Results

After performing pose estimation, you might want to save the results for further analysis or processing. This section will guide you through saving the predicted keypoints and visualizations to your local machine.

To save the predictions in a JSON file, use the `pred_out_dir` argument when running the inferencer:

```python
result_generator = inferencer(img_path, pred_out_dir='predictions')
result = next(result_generator)
```

The predictions will be saved in the `predictions/` folder in JSON format, with each file named after the corresponding input image or video.

For more advanced scenarios, you can also access the predictions directly from the `result` dictionary returned by the inferencer. The key `'predictions'` contains a list of predicted keypoints for each individual instance in the input image or video. You can then manipulate or store these results using your preferred method.

Keep in mind that if you want to save both the visualization images and the prediction files in a single folder, you can use the `out_dir` argument:

```python
result_generator = inferencer(img_path, out_dir='output')
result = next(result_generator)
```

In this case, the visualization images will be saved in the `output/visualization/` folder, while the predictions will be stored in the `output/predictions/` folder.

### Visualization

The inferencer can automatically draw predictions on input images or videos. Visualization results can be displayed in a new window and saved locally.

To view the visualization results in a new window, use the following code:

```python
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
```

Notice that:

- If the input video comes from a webcam, displaying the visualization results in a new window will be enabled by default, allowing users to see the inputs.
- If there is no GUI on the platform, this step may become stuck.

To save the visualization results locally, specify the `vis_out_dir` argument like this:

```python
result_generator = inferencer(img_path, vis_out_dir='vis_results')
result = next(result_generator)
```

The input images or videos with predicted poses will be saved in the `vis_results/` folder.

As seen in the above image, the visualization of estimated poses consists of keypoints (depicted by solid circles) and skeletons (represented by lines). The default size of these visual elements might not produce satisfactory results. Users can adjust the circle size and line thickness using the `radius` and `thickness` arguments, as shown below:

```python
result_generator = inferencer(img_path, show=True, radius=4, thickness=2)
result = next(result_generator)
```

### Arguments of Inferencer

The [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) offers a variety of arguments for customizing pose estimation, visualization, and saving predictions. Below is a list of the arguments available when initializing the inferencer and their descriptions:

| Argument         | Description                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `pose2d`         | Specifies the model alias, configuration file name, or configuration file path for the 2D pose estimation model. |
| `pose2d_weights` | Specifies the URL or local path to the 2D pose estimation model's checkpoint file.                               |
| `pose3d`         | Specifies the model alias, configuration file name, or configuration file path for the 3D pose estimation model. |
| `pose3d_weights` | Specifies the URL or local path to the 3D pose estimation model's checkpoint file.                               |
| `det_model`      | Specifies the model alias, configuration file name, or configuration file path for the object detection model.   |
| `det_weights`    | Specifies the URL or local path to the object detection model's checkpoint file.                                 |
| `det_cat_ids`    | Specifies the list of category IDs corresponding to the object classes to be detected.                           |
| `device`         | The device to perform the inference. If left `None`, the Inferencer will select the most suitable one.           |
| `scope`          | The namespace where the model modules are defined.                                                               |

The inferencer is designed for both visualization and saving predictions. The table below presents the list of arguments available when using the [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24) for inference, along with their compatibility with 2D and 3D inferencing:

| Argument                  | Description                                                                                                                                                       | 2D  | 3D  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- |
| `show`                    | Controls the display of the image or video in a pop-up window.                                                                                                    | ✔️  | ✔️  |
| `radius`                  | Sets the visualization keypoint radius.                                                                                                                           | ✔️  | ✔️  |
| `thickness`               | Determines the link thickness for visualization.                                                                                                                  | ✔️  | ✔️  |
| `kpt_thr`                 | Sets the keypoint score threshold. Keypoints with scores exceeding this threshold will be displayed.                                                              | ✔️  | ✔️  |
| `draw_bbox`               | Decides whether to display the bounding boxes of instances.                                                                                                       | ✔️  | ✔️  |
| `draw_heatmap`            | Decides if the predicted heatmaps should be drawn.                                                                                                                | ✔️  | ❌  |
| `black_background`        | Decides whether the estimated poses should be displayed on a black background.                                                                                    | ✔️  | ❌  |
| `skeleton_style`          | Sets the skeleton style. Options include 'mmpose' (default) and 'openpose'.                                                                                       | ✔️  | ❌  |
| `use_oks_tracking`        | Decides whether to use OKS as a similarity measure in tracking.                                                                                                   | ❌  | ✔️  |
| `tracking_thr`            | Sets the similarity threshold for tracking.                                                                                                                       | ❌  | ✔️  |
| `disable_norm_pose_2d`    | Decides whether to scale the bounding box to the dataset's average bounding box scale and relocate the bounding box to the dataset's average bounding box center. | ❌  | ✔️  |
| `disable_rebase_keypoint` | Decides whether to set the lowest keypoint with height 0.                                                                                                         | ❌  | ✔️  |
| `num_instances`           | Sets the number of instances to visualize in the results. If set to a negative number, all detected instances will be visualized.                                 | ❌  | ✔️  |
| `return_vis`              | Decides whether to include visualization images in the results.                                                                                                   | ✔️  | ✔️  |
| `vis_out_dir`             | Defines the folder path to save the visualization images. If unset, the visualization images will not be saved.                                                   | ✔️  | ✔️  |
| `return_datasamples`      | Determines if the prediction should be returned in the `PoseDataSample` format.                                                                                   | ✔️  | ✔️  |
| `pred_out_dir`            | Specifies the folder path to save the predictions. If unset, the predictions will not be saved.                                                                   | ✔️  | ✔️  |
| `out_dir`                 | If `vis_out_dir` or `pred_out_dir` is unset, these will be set to `f'{out_dir}/visualization'` or `f'{out_dir}/predictions'`, respectively.                       | ✔️  | ✔️  |

### Model Alias

The MMPose library has predefined aliases for several frequently used models. These aliases can be utilized as a shortcut when initializing the [MMPoseInferencer](https://github.com/open-mmlab/mmpose/blob/dev-1.x/mmpose/apis/inferencers/mmpose_inferencer.py#L24), as an alternative to providing the full model configuration name. Here are the available 2D model aliases and their corresponding configuration names:

| Alias     | Configuration Name                                 | Task                            | Pose Estimator | Detector            |
| --------- | -------------------------------------------------- | ------------------------------- | -------------- | ------------------- |
| animal    | rtmpose-m_8xb64-210e_ap10k-256x256                 | Animal pose estimation          | RTMPose-m      | RTMDet-m            |
| human     | rtmpose-m_8xb256-420e_body8-256x192                | Human pose estimation           | RTMPose-m      | RTMDet-m            |
| body26    | rtmpose-m_8xb512-700e_body8-halpe26-256x192        | Human pose estimation           | RTMPose-m      | RTMDet-m            |
| face      | rtmpose-m_8xb256-120e_face6-256x256                | Face keypoint detection         | RTMPose-m      | yolox-s             |
| hand      | rtmpose-m_8xb256-210e_hand5-256x256                | Hand keypoint detection         | RTMPose-m      | ssdlite_mobilenetv2 |
| wholebody | rtmpose-m_8xb64-270e_coco-wholebody-256x192        | Human wholebody pose estimation | RTMPose-m      | RTMDet-m            |
| vitpose   | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-base   | RTMDet-m            |
| vitpose-s | td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192 | Human pose estimation           | ViTPose-small  | RTMDet-m            |
| vitpose-b | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-base   | RTMDet-m            |
| vitpose-l | td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192 | Human pose estimation           | ViTPose-large  | RTMDet-m            |
| vitpose-h | td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-huge   | RTMDet-m            |

The following table lists the available 3D model aliases and their corresponding configuration names:

| Alias   | Configuration Name                           | Task                     | 3D Pose Estimator | 2D Pose Estimator | Detector    |
| ------- | -------------------------------------------- | ------------------------ | ----------------- | ----------------- | ----------- |
| human3d | vid_pl_motionbert_8xb32-120e_h36m            | Human 3D pose estimation | MotionBert        | RTMPose-m         | RTMDet-m    |
| hand3d  | internet_res50_4xb16-20e_interhand3d-256x256 | Hand 3D pose estimation  | InterNet          | -                 | whole image |

In addition, users can utilize the CLI tool to display all available aliases with the following command:

```shell
python demo/inferencer_demo.py --show-alias
```

## Python API: more flexible and customizable

MMPose provides a separate Python API for inference, which is more flexible but requires users to handle inputs and outputs themselves. Therefore, this API is suitable for users who are **familiar with MMPose**.

The Python inference interface provided by MMPose is located in [$MMPOSE/mmpose/apis](https://github.com/open-mmlab/mmpose/tree/dev-1.x/mmpose/apis) directory. Here is an example of building a topdown model and performing inference:

### Build a model

```python
from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

model_cfg = 'configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'

ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth'

device = 'cuda'

# init model
model = init_model(model_cfg, ckpt, device=device)
```

### Inference

```python
img_path = 'tests/data/coco/000000000785.jpg'

# inference on a single image
batch_results = inference_topdown(model, img_path)
```

The inference interface returns a list of PoseDataSample, each of which corresponds to the inference result of an image. The structure of PoseDataSample is as follows:

```python
[
    <PoseDataSample(

        ori_shape: (425, 640)
        img_path: 'tests/data/coco/000000000785.jpg'
        input_size: (192, 256)
        flip_indices: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        img_shape: (425, 640)

        gt_instances: <InstanceData(
                bboxes: array([[  0.,   0., 640., 425.]], dtype=float32)
                bbox_centers: array([[320. , 212.5]], dtype=float32)
                bbox_scales: array([[ 800.    , 1066.6666]], dtype=float32)
                bbox_scores: array([1.], dtype=float32)
            )>

        gt_instance_labels: <InstanceData()>

        pred_instances: <InstanceData(
                keypoints: array([[[365.83333333,  87.50000477],
                            [372.08333333,  79.16667175],
                            [361.66666667,  81.25000501],
                            [384.58333333,  85.41667151],
                            [357.5       ,  85.41667151],
                            [407.5       , 112.50000381],
                            [363.75      , 125.00000334],
                            [438.75      , 150.00000238],
                            [347.08333333, 158.3333354 ],
                            [451.25      , 170.83333492],
                            [305.41666667, 177.08333468],
                            [432.5       , 214.58333325],
                            [401.25      , 218.74999976],
                            [430.41666667, 285.41666389],
                            [370.        , 274.99999762],
                            [470.        , 356.24999452],
                            [403.33333333, 343.74999499]]])
                bbox_scores: array([1.], dtype=float32)
                bboxes: array([[  0.,   0., 640., 425.]], dtype=float32)
                keypoint_scores: array([[0.8720184 , 0.9068178 , 0.89255375, 0.94684595, 0.83111566,
                            0.9929208 , 1.0862956 , 0.9265839 , 0.9781244 , 0.9008082 ,
                            0.9043166 , 1.0150217 , 1.1122335 , 1.0207931 , 1.0099326 ,
                            1.0480015 , 1.0897669 ]], dtype=float32)
                keypoints_visible: array([[0.8720184 , 0.9068178 , 0.89255375, 0.94684595, 0.83111566,
                            0.9929208 , 1.0862956 , 0.9265839 , 0.9781244 , 0.9008082 ,
                            0.9043166 , 1.0150217 , 1.1122335 , 1.0207931 , 1.0099326 ,
                            1.0480015 , 1.0897669 ]], dtype=float32)
            )>
    )>
]
```

You can obtain the predicted keypoints via `.`:

```python
pred_instances = batch_results[0].pred_instances

pred_instances.keypoints
# array([[[365.83333333,  87.50000477],
#         [372.08333333,  79.16667175],
#         [361.66666667,  81.25000501],
#         [384.58333333,  85.41667151],
#         [357.5       ,  85.41667151],
#         [407.5       , 112.50000381],
#         [363.75      , 125.00000334],
#         [438.75      , 150.00000238],
#         [347.08333333, 158.3333354 ],
#         [451.25      , 170.83333492],
#         [305.41666667, 177.08333468],
#         [432.5       , 214.58333325],
#         [401.25      , 218.74999976],
#         [430.41666667, 285.41666389],
#         [370.        , 274.99999762],
#         [470.        , 356.24999452],
#         [403.33333333, 343.74999499]]])
```

### Visualization

In MMPose, most visualizations are implemented based on visualizers. A visualizer is a class that takes a data sample and visualizes it.

MMPose provides a visualizer registry, which users can instantiate using `VISUALIZERS`. Here is an example of using a visualizer to visualize the inference results:

```python
# merge results as a single data sample
results = merge_data_samples(batch_results)

# build the visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)

# set skeleton, colormap and joint connection rule
visualizer.set_dataset_meta(model.dataset_meta)

img = imread(img_path, channel_order='rgb')

# visualize the results
visualizer.add_datasample(
    'result',
    img,
    data_sample=results,
    show=True)
```

MMPose also provides a simpler interface for visualization:

```python
from mmpose.apis import visualize

pred_instances = batch_results[0].pred_instances

keypoints = pred_instances.keypoints
keypoint_scores = pred_instances.keypoint_scores

metainfo = 'config/_base_/datasets/coco.py'

visualize(
    img_path,
    keypoints,
    keypoint_scores,
    metainfo=metainfo,
    show=True)
```
