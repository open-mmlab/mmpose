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
<summary align="right"><a href="https://arxiv.org/abs/2303.16160">UBody (CVPR'2023)</a></summary>

```bibtex
@article{lin2023one,
  title={One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer},
  author={Lin, Jing and Zeng, Ailing and Wang, Haoqian and Zhang, Lei and Li, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```

</details>

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [pose_hrnet_w32](/configs/wholebody_2d_keypoint/topdown_heatmap/ubody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.685  |  0.759  |  0.564  |  0.675  |  0.625  |  0.705  |  0.516  |  0.609  |  0.549   |  0.646   | [ckpt](https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/ubody/td-hm_hrnet-w32_8xb64-210e_ubody-coco-256x192-7c227391_20230807.pth) | [log](https://download.openmmlab.com/mmpose/v1/wholebody_2d_keypoint/ubody/td-hm_hrnet-w32_8xb64-210e_ubody-coco-256x192-7c227391_20230807.json) |
