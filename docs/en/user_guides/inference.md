# Inference with existing models

MMPose provides a wide variety of pre-trained models for pose estimation, which can be found in the [Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html).
This guide will demonstrate **how to perform inference**, or running pose estimation on provided images or videos using trained models.

For instructions on testing existing models on standard datasets, refer to this [guide](./train_and_test.md#test).

In MMPose, a model is defined by a configuration file, while its pre-existing parameters are stored in a checkpoint file. You can find the model configuration files and corresponding checkpoint URLs in the [Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html). We recommend starting with the HRNet model, using [this configuration file](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) and [this checkpoint file](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth).

## Inferencer: a Unified Inference Interface

MMPose offers a comprehensive API for inference, known as `MMPoseInferencer`. This API enables users to perform inference on both images and videos using all the models supported by MMPose. Furthermore, the API provides automatic visualization of inference results and allows for the convenient saving of predictions.

### Basic Usage

The `MMPoseInferencer` can be used in any Python program to perform pose estimation. Below is an example of inference on a given image using the pre-trained human pose estimator within the Python shell.

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

The predictions will be save in `predictions/000000000785.json`. The argument names correspond with the `MMPoseInferencer`, which serves as an API.

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

The `MMPoseInferencer` offers a variety of arguments for customizing pose estimation, visualization, and saving predictions. Below is a list of the arguments available when initializing the inferencer and their descriptions:

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

The inferencer is designed for both visualization and saving predictions. The table below presents the list of arguments available when using the `MMPoseInferencer` for inference, along with their compatibility with 2D and 3D inferencing:

| Argument                 | Description                                                                                                                                                       | 2D  | 3D  |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- |
| `show`                   | Controls the display of the image or video in a pop-up window.                                                                                                    | ✔️  | ✔️  |
| `radius`                 | Sets the visualization keypoint radius.                                                                                                                           | ✔️  | ✔️  |
| `thickness`              | Determines the link thickness for visualization.                                                                                                                  | ✔️  | ✔️  |
| `kpt_thr`                | Sets the keypoint score threshold. Keypoints with scores exceeding this threshold will be displayed.                                                              | ✔️  | ✔️  |
| `draw_bbox`              | Decides whether to display the bounding boxes of instances.                                                                                                       | ✔️  | ✔️  |
| `draw_heatmap`           | Decides if the predicted heatmaps should be drawn.                                                                                                                | ✔️  | ❌  |
| `black_background`       | Decides whether the estimated poses should be displayed on a black background.                                                                                    | ✔️  | ❌  |
| `skeleton_style`         | Sets the skeleton style. Options include 'mmpose' (default) and 'openpose'.                                                                                       | ✔️  | ❌  |
| `use_oks_tracking`       | Decides whether to use OKS as a similarity measure in tracking.                                                                                                   | ❌  | ✔️  |
| `tracking_thr`           | Sets the similarity threshold for tracking.                                                                                                                       | ❌  | ✔️  |
| `norm_pose_2d`           | Decides whether to scale the bounding box to the dataset's average bounding box scale and relocate the bounding box to the dataset's average bounding box center. | ❌  | ✔️  |
| `rebase_keypoint_height` | Decides whether to set the lowest keypoint with height 0.                                                                                                         | ❌  | ✔️  |
| `return_vis`             | Decides whether to include visualization images in the results.                                                                                                   | ✔️  | ✔️  |
| `vis_out_dir`            | Defines the folder path to save the visualization images. If unset, the visualization images will not be saved.                                                   | ✔️  | ✔️  |
| `return_datasample`      | Determines if the prediction should be returned in the `PoseDataSample` format.                                                                                   | ✔️  | ✔️  |
| `pred_out_dir`           | Specifies the folder path to save the predictions. If unset, the predictions will not be saved.                                                                   | ✔️  | ✔️  |
| `out_dir`                | If `vis_out_dir` or `pred_out_dir` is unset, these will be set to `f'{out_dir}/visualization'` or `f'{out_dir}/predictions'`, respectively.                       | ✔️  | ✔️  |

### Model Alias

The MMPose library has predefined aliases for several frequently used models. These aliases can be utilized as a shortcut when initializing the `MMPoseInferencer`, as an alternative to providing the full model configuration name. Here are the available 2D model aliases and their corresponding configuration names:

| Alias     | Configuration Name                                 | Task                            | Pose Estimator | Detector            |
| --------- | -------------------------------------------------- | ------------------------------- | -------------- | ------------------- |
| animal    | rtmpose-m_8xb64-210e_ap10k-256x256                 | Animal pose estimation          | RTMPose-m      | RTMDet-m            |
| human     | rtmpose-m_8xb256-420e_aic-coco-256x192             | Human pose estimation           | RTMPose-m      | RTMDet-m            |
| face      | rtmpose-m_8xb64-60e_wflw-256x256                   | Face keypoint detection         | RTMPose-m      | yolox-s             |
| hand      | rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256   | Hand keypoint detection         | RTMPose-m      | ssdlite_mobilenetv2 |
| wholebody | rtmpose-m_8xb64-270e_coco-wholebody-256x192        | Human wholebody pose estimation | RTMPose-m      | RTMDet-m            |
| vitpose   | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-base   | RTMDet-m            |
| vitpose-s | td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192 | Human pose estimation           | ViTPose-small  | RTMDet-m            |
| vitpose-b | td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-base   | RTMDet-m            |
| vitpose-l | td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192 | Human pose estimation           | ViTPose-large  | RTMDet-m            |
| vitpose-h | td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192  | Human pose estimation           | ViTPose-huge   | RTMDet-m            |

The following table lists the available 3D model aliases and their corresponding configuration names:

| Alias   | Configuration Name                                        | Task                     | 3D Pose Estimator | 2D Pose Estimator | Detector |
| ------- | --------------------------------------------------------- | ------------------------ | ----------------- | ----------------- | -------- |
| human3d | pose-lift_videopose3d-243frm-supv-cpn-ft_8xb128-200e_h36m | Human 3D pose estimation | VideoPose3D       | RTMPose-m         | RTMDet-m |

In addition, users can utilize the CLI tool to display all available aliases with the following command:

```shell
python demo/inferencer_demo.py --show-alias
```
