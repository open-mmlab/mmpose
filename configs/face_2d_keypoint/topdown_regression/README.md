# Top-down regression-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. At the 2nd stage, regression based methods directly regress the keypoint coordinates given the features extracted from the bounding box area, following the paradigm introduced in [Deeppose: Human pose estimation via deep neural networks](http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146515040-a82a8a29-d6bc-42f1-a2ab-7dfa610ce363.png">
</div>

## Results and Models

### WFLW Dataset

Result on WFLW test set

| Model                                                           | Input Size | NME  |                              ckpt                              |                              log                              |
| :-------------------------------------------------------------- | :--------: | :--: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [ResNet-50](/configs/face_2d_keypoint/topdown_regression/wflw/td-reg_res50_8xb64-210e_wflw-256x256.py) |  256x256   | 4.88 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256-92d0ba7f_20210303.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_20210303.log.json) |
| [ResNet-50+WingLoss](/configs/face_2d_keypoint/topdown_regression/wflw/td-reg_res50_wingloss_8xb64-210e_wflw-256x256.py) |  256x256   | 4.67 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss-f82a5e53_20210303.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_wingloss_20210303.log.json) |
| [ResNet-50+SoftWingLoss](/configs/face_2d_keypoint/topdown_regression/wflw/td-reg_res50_softwingloss_8xb64-210e_wflw-256x256.py) |  256x256   | 4.44 | [ckpt](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_softwingloss-4d34f22a_20211212.pth) | [log](https://download.openmmlab.com/mmpose/face/deeppose/deeppose_res50_wflw_256x256_softwingloss_20211212.log.json) |
