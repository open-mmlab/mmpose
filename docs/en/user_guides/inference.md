# Inference with existing models

MMPose provides hundreds of pre-trained models for pose estimation in [Model Zoo](https://mmpose.readthedocs.io/en/1.x/modelzoo.html).
This note will show **how to perform inference**, which means running pose estimation on given images or videos with trained models.

As for how to test existing models on standard datasets, please see this [guide](./train_and_test.md#test).

In MMPose, a model is defined by a configuration file and existing model parameters are saved in a checkpoint file.

To start with, we recommend HRNet model with [this configuration file](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) and [this checkpoint file](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth). It is recommended to download the checkpoint file to `checkpoints` directory.

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
