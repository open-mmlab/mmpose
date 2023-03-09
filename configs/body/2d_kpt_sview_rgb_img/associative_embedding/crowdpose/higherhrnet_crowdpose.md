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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

</details>

Results on CrowdPose test without multi-scale test

| Arch                                           | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) |                      ckpt                      |                      log                      |
| :--------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :----: | :----: | :----: | :--------------------------------------------: | :-------------------------------------------: |
| [HigherHRNet-w32](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/crowdpose/higherhrnet_w32_crowdpose_512x512.py) |  512x512   | 0.655 |      0.859      |      0.705      | 0.728  | 0.660  | 0.577  | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512-1aa4a132_20201017.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512_20201017.log.json) |

Results on CrowdPose test with multi-scale test. 2 scales (\[2, 1\]) are used

| Arch                                           | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) |                      ckpt                      |                      log                      |
| :--------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :----: | :----: | :----: | :--------------------------------------------: | :-------------------------------------------: |
| [HigherHRNet-w32](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/crowdpose/higherhrnet_w32_crowdpose_512x512.py) |  512x512   | 0.661 |      0.864      |      0.710      | 0.742  | 0.670  | 0.566  | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512-1aa4a132_20201017.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512_20201017.log.json) |
