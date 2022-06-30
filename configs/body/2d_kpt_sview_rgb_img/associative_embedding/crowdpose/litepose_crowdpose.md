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
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Lite_Pose_Efficient_Architecture_Design_for_2D_Human_Pose_Estimation_CVPR_2022_paper.html">LitePose (CVPR'2022)</a></summary>

```bibtex
@inproceedings{wang2022lite,
  title={Lite pose: Efficient architecture design for 2d human pose estimation},
  author={Wang, Yihan and Li, Muyang and Cai, Han and Chen, Wei-Ming and Han, Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13126--13136},
  year={2022}
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

| Arch                                                                                                               | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |    ckpt    |    log    |
| :----------------------------------------------------------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--------: | :-------: |
| [LitePose-XS](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/crowdpose/litepose_XS_crowdpose_256x256.py) |  256x256   | 0.494 |      0.746      |      0.508      | 0.555 |      0.785      | [ckpt](<>) | [log](<>) |
| [LitePose-S](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/crowdpose/litepose_S_crowdpose_448x448.py)   |  448x448   |   0   |        0        |        0        |   0   |        0        | [ckpt](<>) | [log](<>) |
