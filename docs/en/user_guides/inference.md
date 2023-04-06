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

# create the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
```

If everything works fine, you will see the following image in a new window:
![inferencer_result_coco](https://user-images.githubusercontent.com/26127467/220008302-4a57fd44-0978-408e-8351-600e5513316a.jpg)

The variable `result` is a dictionary that contains two keys, `'visualization'` and `'predictions'`. The `'visualization'` key is meant to store visualization results, but since the `return_vis` argument wasn't specified, this list remains empty. The `'predictions'` key, however, holds a list of estimated keypoints for each detected instance.

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
| `det_model`      | Specifies the model alias, configuration file name, or configuration file path for the object detection model.   |
| `det_weights`    | Specifies the URL or local path to the object detection model's checkpoint file.                                 |
| `det_cat_ids`    | Specifies the list of category IDs corresponding to the object classes to be detected.                           |
| `device`         | The device to perform the inference. If left `None`, the Inferencer will select the most suitable one.           |
| `scope`          | The namespace where the model modules are defined.                                                               |

The inferencer is designed to handle both visualization and saving of predictions. Here is a list of arguments available when performing inference with the `MMPoseInferencer`:

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

### Model Alias

MMPose provides a set of pre-defined aliases for commonly used models. These aliases can be used as shorthand when initializing the `MMPoseInferencer` instead of specifying the full model configuration name. Below is a list of the available model aliases and their corresponding configuration names:

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

In addition, users can utilize the CLI tool to display all available aliases with the following command:

```shell
python demo/inferencer_demo.py --show-alias
```
