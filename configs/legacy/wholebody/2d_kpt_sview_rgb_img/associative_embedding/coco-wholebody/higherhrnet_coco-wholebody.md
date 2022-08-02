<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.05424">Associative Embedding (NIPS'2017)</a></summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_HigherHRNet_Scale-Aware_Representation_Learning_for_Bottom-Up_Human_Pose_Estimation_CVPR_2020_paper.html">HigherHRNet (CVPR'2020)</a></summary>

```bibtex
@inproceedings{cheng2020higherhrnet,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Cheng, Bowen and Xiao, Bin and Wang, Jingdong and Shi, Honghui and Huang, Thomas S and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5386--5395},
  year={2020}
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

Results on COCO-WholeBody v1.0 val  without multi-scale test

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [HigherHRNet-w32+](/configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/higherhrnet_w32_coco_wholebody_512x512.py)  | 512x512 | 0.590 | 0.672 | 0.185 | 0.335 | 0.676 | 0.721 | 0.212 | 0.298 | 0.401 | 0.493 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_wholebody_512x512_plus-2fa137ab_20210517.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_wholebody_512x512_plus_20210517.log.json) |
| [HigherHRNet-w48+](/configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/higherhrnet_w48_coco_wholebody_512x512.py)  | 512x512 | 0.630 | 0.706 | 0.440 | 0.573 | 0.730 | 0.777 | 0.389 | 0.477 | 0.487 | 0.574 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_wholebody_512x512_plus-934f08aa_20210517.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_wholebody_512x512_plus_20210517.log.json) |

Note: `+` means the model is first pre-trained on original COCO dataset, and then fine-tuned on COCO-WholeBody dataset. We find this will lead to better performance.
