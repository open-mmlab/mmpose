<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html">SimpleBaseline2D (ECCV'2018)</a></summary>

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
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
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/posetrack18/td-hm_res50_8xb64-20e_posetrack18-256x192.py) |  256x192   | 86.5 | 87.7 | 82.5 | 75.8 | 80.1 | 78.8 | 74.2 | 81.2  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192-a62807c7_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.
