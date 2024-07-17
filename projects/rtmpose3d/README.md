# RTMPose3D: Real-Time 3D Pose Estimation toolkit based on RTMPose

> ***Technical Report***:
> [RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation](https://arxiv.org/abs/2407.08634)

## Abstract

RTMPose3D is a toolkit for real-time 3D pose estimation. It is based on the RTMPose model, which is a 2D pose estimation model that is capable of predicting 2D keypoints and body part associations in real-time. RTMPose3D extends RTMPose by adding a 3D pose estimation branch that can predict 3D keypoints from images directly.

Please refer to our [technical report](https://arxiv.org/pdf/2407.08634) for more details.

## 🗂️ Model Zoo

| Model                                                      | AP on COCO-Wholebody | MPJPE on H3WB |                                                   Download                                                    |
| :--------------------------------------------------------- | :------------------: | :-----------: | :-----------------------------------------------------------------------------------------------------------: |
| [RTMW3D-L](./configs/rtmw3d-l_8xb64_cocktail14-384x288.py) |        0.678         |     0.056     | [ckpt](https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth) |
| [RTMW3D-X](./configs/rtmw3d-x_8xb32_cocktail14-384x288.py) |        0.680         |     0.057     | [ckpt](https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth) |

## 📚 Usage

👉🏼 TRY RTMPose3D NOW

```bash
cd /path/to/mmpose/projects/rtmpose3d
export PYTHONPATH=$(pwd):$PYTHONPATH
python body3d_img2pose_demo.py configs/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs\rtmw3d-l_8xb64_cocktail14-384x288.py rtmw3d-l_cock14-0d4ad840_20240422.pth --input /path/to/image --output-root /path/to/output
```

## 📜 Citation [🔝](#-table-of-contents)

If you find RTMPose3D toolkit or RTMW3D models useful in your research, please consider cite:

```bibtex
@article{jiang2024rtmw,
  title={RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation},
  author={Jiang, Tao and Xie, Xinchen and Li, Yining},
  journal={arXiv preprint arXiv:2407.08634},
  year={2024}
}

@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
