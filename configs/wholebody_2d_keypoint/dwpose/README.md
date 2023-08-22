# DWPose

Whole-body pose estimation localizes the human body, hand, face, and foot keypoints in an image. This task is challenging due to multi-scale body parts, fine-grained localization for low-resolution regions, and data scarcity. Meanwhile, applying a highly efficient and accurate pose estimator to widely human-centric understanding and generation tasks is urgent. In this work, we present a two-stage pose **D**istillation for **W**hole-body **P**ose estimators, named **DWPose**, to improve their effectiveness and efficiency. The first-stage distillation designs a weight-decay strategy while utilizing a teacher's intermediate feature and final logits with both visible and invisible keypoints to supervise the student from scratch. The second stage distills the student model itself to further improve performance. Different from the previous self-knowledge distillation, this stage finetunes the student's head with only 20% training time as a plug-and-play training strategy. For data limitations, we explore the UBody dataset that contains diverse facial expressions and hand gestures for real-life applications. Comprehensive experiments show the superiority of our proposed simple yet effective methods. We achieve new state-of-the-art performance on COCO-WholeBody, significantly boosting the whole-body AP of RTMPose-l from 64.8% to 66.5%, even surpassing RTMPose-x teacher with 65.3% AP. We release a series of models with different sizes, from tiny to large, for satisfying various downstream tasks.

## Results and Models

### COCO-WholeBody Dataset

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

- DWPose Models are supported by [DWPose](https://github.com/IDEA-Research/DWPose)
- Models are trained and distilled on the following datasets:
  - [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/)
  - [UBody](https://github.com/IDEA-Research/OSX)

| Config       |    S1 Dis_config    |    S2 Dis_config    | Input Size | Whole AP | Whole AR | FLOPS<sup><br>(G) | ORT-Latency<sup><br>(ms)<sup><br>(i7-11700) | TRT-FP16-Latency<sup><br>(ms)<sup><br>(GTX 1660Ti) |    Download    |
| :----------- | :-----------------: | :-----------------: | :--------: | :------: | :------: | :---------------: | :-----------------------------------------: | :------------------------------------------------: | :------------: |
| [DWPose-t](../rtmpose/ubody/rtmpose-t_8xb64-270e_coco-ubody-wholebody-256x192.py) | [DW l-t](../dwpose/ubody/s1_dis/dwpose_l_dis_t_coco-ubody-256x192.py) | [DW t-t](../dwpose/ubody/s2_dis/dwpose_t-tt_coco-ubody-256x192.py) |  256x192   |   48.5   |   58.4   |        0.5        |                      -                      |                         -                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth) |
| [DWPose-s](../rtmpose/ubody/rtmpose-s_8xb64-270e_coco-ubody-wholebody-256x192.py) | [DW l-s](../dwpose/ubody/s1_dis/dwpose_l_dis_s_coco-ubody-256x192.py) | [DW s-s](../dwpose/ubody/s2_dis/dwpose_s-ss_coco-ubody-256x192.py) |  256x192   |   53.8   |   63.2   |        0.9        |                      -                      |                         -                          | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth) |
| [DWPose-m](../rtmpose/ubody/rtmpose-m_8xb64-270e_coco-ubody-wholebody-256x192.py) | [DW l-m](../dwpose/ubody/s1_dis/dwpose_l_dis_m_coco-ubody-256x192.py) | [DW m-m](../dwpose/ubody/s2_dis/dwpose_m-mm_coco-ubody-256x192.py) |  256x192   |   60.6   |   69.5   |       2.22        |                    13.50                    |                        4.00                        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth) |
| [DWPose-l](../rtmpose/ubody/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py) | [DW x-l](../dwpose/ubody/s1_dis/dwpose_x_dis_l_coco-ubody-256x192.py) | [DW l-l](../dwpose/ubody/s2_dis/dwpose_l-ll_coco-ubody-256x192.py) |  256x192   |   63.1   |   71.7   |       4.52        |                    23.41                    |                        5.67                        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth) |
| [DWPose-l](../rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py) | [DW x-l](../dwpose/ubody/s1_dis/dwpose_x_dis_l_coco-ubody-384x288.py) | [DW l-l](../dwpose/ubody/s2_dis/dwpose_l-ll_coco-ubody-384x288.py) |  384x288   |   66.5   |   74.3   |       10.07       |                    44.58                    |                        7.68                        | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.pth) |

## Train a model

### Train DWPose with the first stage distillation

```
bash tools/dist_train.sh configs/wholebody_2d_keypoint/dwpose/ubody/s1_dis/rtmpose_x_dis_l_coco-ubody-384x288.py 8
```

### Tansfer the S1 distillation models into regular models

```
# first stage distillation
python pth_transfer.py $dis_ckpt $new_pose_ckpt
```

⭐Before S2 distillation, you should add your model path into 'teacher_pretrained' of your S2 dis_config.

### Train DWPose with the second stage distillation

```
bash tools/dist_train.sh configs/wholebody_2d_keypoint/dwpose/ubody/s2_dis/dwpose_l-ll_coco-ubody-384x288.py 8
```

### Tansfer the S2 distillation models into regular models

```
# second stage distillation
python pth_transfer.py $dis_ckpt $new_pose_ckpt --two_dis
```

## Citation

```
@article{yang2023effective,
  title={Effective Whole-body Pose Estimation with Two-stages Distillation},
  author={Yang, Zhendong and Zeng, Ailing and Yuan, Chun and Li, Yu},
  journal={arXiv preprint arXiv:2307.15880},
  year={2023}
}
```
