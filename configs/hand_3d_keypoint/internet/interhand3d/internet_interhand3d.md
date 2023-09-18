<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/content/pdf/10.1007/978-3-030-58565-5_33.pdf">InterNet (ECCV'2020)</a></summary>

```bibtex
@InProceedings{Moon_2020_ECCV_InterHand2.6M,
author = {Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},
title = {InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/content/pdf/10.1007/978-3-030-58565-5_33.pdf">InterHand2.6M (ECCV'2020)</a></summary>

```bibtex
@InProceedings{Moon_2020_ECCV_InterHand2.6M,
author = {Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},
title = {InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

</details>

Results on InterHand2.6M val & test set

| Train Set | Set       |                    Arch                    | Input Size | MPJPE-single | MPJPE-interacting | MPJPE-all | MRRPE | APh  |                    ckpt                    |                    log                    |
| :-------- | :-------- | :----------------------------------------: | :--------: | :----------: | :---------------: | :-------: | :---: | :--: | :----------------------------------------: | :---------------------------------------: |
| All       | test(H+M) | [InterNet_resnet_50](/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py) |  256x256   |     9.69     |       13.72       |   11.86   | 29.27 | 0.99 | [ckpt](https://download.openmmlab.com/mmpose/v1/hand_3d_keypoint/internet/interhand3d/internet_res50_interhand3d-d6ff20d6_20230913.pth) | [log](https://download.openmmlab.com/mmpose/v1/hand_3d_keypoint/internet/interhand3d/internet_res50_interhand3d-d6ff20d6_20230913.json) |
| All       | val(M)    | [InterNet_resnet_50](/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py) |  256x256   |    11.30     |       15.57       |   13.36   | 32.15 | 0.98 | [ckpt](https://download.openmmlab.com/mmpose/v1/hand_3d_keypoint/internet/interhand3d/internet_res50_interhand3d-d6ff20d6_20230913.pth) | [log](https://download.openmmlab.com/mmpose/v1/hand_3d_keypoint/internet/interhand3d/internet_res50_interhand3d-d6ff20d6_20230913.json) |
| All       | test(H+M) | [InterNet_resnet_50\*](/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py) |  256x256   |     9.47     |       13.40       |   11.59   | 29.28 | 0.99 | [ckpt](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth) | [log](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256_20210702.log.json) |
| All       | val(M)    | [InterNet_resnet_50\*](/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py) |  256x256   |    11.22     |       15.23       |   13.16   | 31.73 | 0.98 | [ckpt](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth) | [log](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256_20210702.log.json) |

*Models with * are trained in [MMPose 0.x](https://github.com/open-mmlab/mmpose/tree/0.x). The checkpoints and logs are only for validation.*
