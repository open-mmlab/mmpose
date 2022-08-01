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

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [pose_resnet_50](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_256x192.py)  | 256x192 | 0.652 | 0.739 | 0.614 | 0.746 | 0.608 | 0.716 | 0.460 | 0.584 | 0.520 | 0.633 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192-9e37ed88_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_50](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_384x288.py)  | 384x288 | 0.666 | 0.747 | 0.635 | 0.763 | 0.732 | 0.812 | 0.537 | 0.647 | 0.573 | 0.671 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288-ce11e294_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_101](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res101_coco_wholebody_256x192.py)  | 256x192 | 0.670 | 0.754 | 0.640 | 0.767 | 0.611 | 0.723 | 0.463 | 0.589 | 0.533 | 0.647 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192-7325f982_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_101](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res101_coco_wholebody_384x288.py)  | 384x288 | 0.692 | 0.770 | 0.680 | 0.798 | 0.747 | 0.822 | 0.549 | 0.658 | 0.597 | 0.692 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288-6c137b9a_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_152](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res152_coco_wholebody_256x192.py)  | 256x192 | 0.682 | 0.764 | 0.662 | 0.788 | 0.624 | 0.728 | 0.482 | 0.606 | 0.548 | 0.661 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192-5de8ae23_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_152](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res152_coco_wholebody_384x288.py)  | 384x288 | 0.703 | 0.780 | 0.693 | 0.813 | 0.751 | 0.825 | 0.559 | 0.667 | 0.610 | 0.705 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288-eab8caa8_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288_20201004.log.json) |
