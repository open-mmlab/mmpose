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
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/hrnet_w32_ochuman_256x192.py)  | 256x192 | 0.591 | 0.748 | 0.641 | 0.631 | 0.775 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_20200708.log.json) |
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/hrnet_w32_ochuman_384x288.py)  | 384x288 | 0.606 | 0.748 | 0.650 | 0.647 | 0.776 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288-d9f0d786_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_20200708.log.json) |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/hrnet_w48_ochuman_256x192.py) | 256x192 | 0.611 | 0.752 | 0.663 | 0.648 | 0.778 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_20200708.log.json) |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/hrnet_w48_ochuman_384x288.py) | 384x288 | 0.616 | 0.749 | 0.663 | 0.653 | 0.773 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_20200708.log.json) |
