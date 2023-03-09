<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.html">PoseTrack18 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{andriluka2018posetrack,
  title={Posetrack: A benchmark for human pose estimation and tracking},
  author={Andriluka, Mykhaylo and Iqbal, Umar and Insafutdinov, Eldar and Pishchulin, Leonid and Milan, Anton and Gall, Juergen and Schiele, Bernt},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5167--5176},
  year={2018}
}
```

</details>

Results on PoseTrack2018 val with ground-truth bounding boxes

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w32_posetrack18_256x192.py) |  256x192   | 87.4 | 88.6 | 84.3 | 78.5 | 79.7 | 81.8 | 78.8 | 83.0  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192-1ee951c4_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192_20201028.log.json) |
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w32_posetrack18_384x288.py) |  384x288   | 87.0 | 88.8 | 85.0 | 80.1 | 80.5 | 82.6 | 79.4 | 83.6  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288-806f00a3_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288_20211130.log.json) |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w48_posetrack18_256x192.py) |  256x192   | 88.2 | 90.1 | 85.8 | 80.8 | 80.7 | 83.3 | 80.3 | 84.4  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192-b5d9b3f1_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192_20211130.log.json) |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w48_posetrack18_384x288.py) |  384x288   | 87.8 | 90.0 | 85.9 | 81.3 | 81.1 | 83.3 | 80.9 | 84.5  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288-5fd6d3ff_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288_20211130.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

Results on PoseTrack2018 val with [MMDetection](https://github.com/open-mmlab/mmdetection) pre-trained [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) (X-101-64x4d-FPN) human detector

| Arch                                                 | Input Size | Head | Shou | Elb  | Wri  | Hip  | Knee | Ankl | Total |                         ckpt                          |                         log                          |
| :--------------------------------------------------- | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :---------------------------------------------------: | :--------------------------------------------------: |
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w32_posetrack18_256x192.py) |  256x192   | 78.0 | 82.9 | 79.5 | 73.8 | 76.9 | 76.6 | 70.2 | 76.9  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192-1ee951c4_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192_20201028.log.json) |
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w32_posetrack18_384x288.py) |  384x288   | 79.9 | 83.6 | 80.4 | 74.5 | 74.8 | 76.1 | 70.5 | 77.3  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288-806f00a3_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_384x288_20211130.log.json) |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w48_posetrack18_256x192.py) |  256x192   | 80.1 | 83.4 | 80.6 | 74.8 | 74.3 | 76.8 | 70.4 | 77.4  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192-b5d9b3f1_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192_20211130.log.json) |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/posetrack18/hrnet_w48_posetrack18_384x288.py) |  384x288   | 80.2 | 83.8 | 80.9 | 75.2 | 74.7 | 76.7 | 71.7 | 77.8  | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288-5fd6d3ff_20211130.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_384x288_20211130.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.
