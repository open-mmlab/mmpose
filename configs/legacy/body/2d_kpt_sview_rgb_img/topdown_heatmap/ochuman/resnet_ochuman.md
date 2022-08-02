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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Pose2Seg_Detection_Free_Human_Instance_Segmentation_CVPR_2019_paper.html">OCHuman (CVPR'2019)</a></summary>

```bibtex
@inproceedings{zhang2019pose2seg,
  title={Pose2seg: Detection free human instance segmentation},
  author={Zhang, Song-Hai and Li, Ruilong and Dong, Xin and Rosin, Paul and Cai, Zixi and Han, Xi and Yang, Dingcheng and Huang, Haozhi and Hu, Shi-Min},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={889--898},
  year={2019}
}
```

</details>

Results on OCHuman test dataset with ground-truth bounding boxes

Following the common setting, the models are trained on COCO train dataset, and evaluate on OCHuman dataset.

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py)  | 256x192 | 0.546 | 0.726 | 0.593 | 0.592 | 0.755 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_20200709.log.json) |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_384x288.py)  | 384x288 | 0.539 | 0.723 | 0.574 | 0.588 | 0.756 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288-e6f795e9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_20200709.log.json) |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192.py) | 256x192 | 0.559 | 0.724 | 0.606 | 0.605 | 0.751 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_20200708.log.json) |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_384x288.py) | 384x288 | 0.571 | 0.715 | 0.615 | 0.615 | 0.748 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288-8c71bdc9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_20200709.log.json) |
| [pose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192.py) | 256x192 | 0.570 | 0.725 | 0.617 | 0.616 | 0.754 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_20200709.log.json) |
| [pose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_384x288.py) | 384x288 | 0.582 | 0.723 | 0.627 | 0.627 | 0.752 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_20200709.log.json) |
