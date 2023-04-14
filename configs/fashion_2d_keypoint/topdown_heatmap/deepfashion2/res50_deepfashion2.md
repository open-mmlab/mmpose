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
<summary align="right"><a href="https://arxiv.org/pdf/1901.07973.pdf">DeepFashion2 (CVPR'2019)</a></summary>

```bibtex
@article{DeepFashion2,
  author = {Yuying Ge and Ruimao Zhang and Lingyun Wu and Xiaogang Wang and Xiaoou Tang and Ping Luo},
  title={A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images},
  journal={CVPR},
  year={2019}
}
```

</details>

Results on DeepFashion2 val set

| Set                   |                        Arch                         | Input Size | PCK@0.2 |  AUC  |  EPE  |                        ckpt                         |                        log                         |
| :-------------------- | :-------------------------------------------------: | :--------: | :-----: | :---: | :---: | :-------------------------------------------------: | :------------------------------------------------: |
| short_sleeved_shirt   | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-60e_deepfashion2_short_sleeved_shirt_256x192.py) |  256x256   |  0.983  | 0.680 | 11.75 | [ckpt](https://drive.google.com/file/d/10BVvn7F2NmhNCwhVWGzXqb89YqJDj64v/view?usp=share_link) | [log](https://drive.google.com/file/d/1126pV_gN9tTEAfAsBDQuapo5s4wg2Xpr/view?usp=share_link) |
| long_sleeved_shirt    | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-120e_deepfashion2_long_sleeved_shirt_256x192.py) |  256x256   |  0.961  | 0.535 | 20.23 | [ckpt](https://drive.google.com/file/d/1eJmJ9VUlg7t68tTY5Wud8fWpPQ8E6z2p/view?usp=share_link) | [log](https://drive.google.com/file/d/1-w_NpZonD0z3-un_uj86OHPwROre--oq/view?usp=share_link) |
| short_sleeved_outwear | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb8-150e_deepfashion2_short_sleeved_outwear_256x192.py) |  256x256   |  0.933  | 0.486 | 22.63 | [ckpt](https://drive.google.com/file/d/1a-veNt2qUC2MPeWJxeOxHYWHrEHCx2QO/view?usp=share_link) | [log](https://drive.google.com/file/d/1lFw3w4VPz9wlDCfqWvmys4IHijWJ4xHh/view?usp=share_link) |
| long_sleeved_outwear  | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb16-120e_deepfashion2_long_sleeved_outwear_256x192.py) |  256x256   |  0.980  | 0.488 | 19.89 | [ckpt](https://drive.google.com/file/d/1_939tgYISZAyUkx7kEAz6aauARKa5KwL/view?usp=share_link) | [log](https://drive.google.com/file/d/1auEFCmJ5-Px5qfaicOEWR0zkmusWR0wb/view?usp=share_link) |
| vest                  | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-120e_deepfashion2_vest_256x192.py) |  256x256   |  0.940  | 0.609 | 20.32 | [ckpt](https://drive.google.com/file/d/1HneXiaukHqbSZGRUK8dc10U9phlFzxD0/view?usp=share_link) | [log](https://drive.google.com/file/d/1s1FIVxp5AcUeBmjHAedi8cMuQ2lzbN3L/view?usp=share_link) |
| sling                 | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb32-120e_deepfashion2_sling_256x192.py) |  256x256   |  0.957  | 0.453 | 24.41 | [ckpt](https://drive.google.com/file/d/1K5zmkbXV3VsLwjyY5BFW2nAho5DWoV9k/view?usp=share_link) | [log](https://drive.google.com/file/d/17t-L_emHzfq0PiYGARaz7YlG3FnlpAj4/view?usp=share_link) |
| shorts                | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-210e_deepfashion2_shorts_256x192.py) |  256x256   |  0.961  | 0.662 | 14.45 | [ckpt](https://drive.google.com/file/d/1DjetHnrHPK03cGAIGDXgRo1pKYC8kW2B/view?usp=share_link) | [log](https://drive.google.com/file/d/14T3cbEeNzbUH4G8cAZROl_ncY62o8yW7/view?usp=share_link) |
| trousers              | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-60e_deepfashion2_trousers_256x192.py) |  256x256   |  0.966  | 0.600 | 16.09 | [ckpt](https://drive.google.com/file/d/1NzeVj9N158VU25C_MDIBtcCVK1Q1HIxY/view?usp=share_link) | [log](https://drive.google.com/file/d/1AWZcGNcLsB4qDiXFnmOnWpqetDN7lIJu/view?usp=share_link) |
| skirt                 | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-120e_deepfashion2_skirt_256x192.py) |  256x256   |  0.944  | 0.634 | 18.80 | [ckpt](https://drive.google.com/file/d/1JP5E4juETQmni9J2xPZKl8EkbRyp8gx_/view?usp=share_link) | [log](https://drive.google.com/file/d/1TjQw4y570bUDbOFkJ-wuDWUIBSctf-GL/view?usp=share_link) |
| short_sleeved_dress   | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-150e_deepfashion2_short_sleeved_dress_256x192.py) |  256x256   |  0.977  | 0.578 | 16.77 | [ckpt](https://drive.google.com/file/d/1-8lk3YJS8SZ3B0-EtRBBXkPJcNp9OJJn/view?usp=share_link) | [log](https://drive.google.com/file/d/1OoZvq6GtdcI0aEiVkdOWcm8yJbIsE3kx/view?usp=share_link) |
| long_sleeved_dress    | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb16-150e_deepfashion2_long_sleeved_dress_256x192.py) |  256x256   |  0.959  | 0.484 | 24.06 | [ckpt](https://drive.google.com/file/d/1F6qq8A3h5Pvx3k-wAXRBA2VL4JUBPNWn/view?usp=share_link) | [log](https://drive.google.com/file/d/1w4AVD7VW4PTeN0Pb5qXrbqnSMY2PMkMt/view?usp=share_link) |
| vest_dress            | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-150e_deepfashion2_vest_dress_256x192.py) |  256x256   |  0.968  | 0.568 | 17.80 | [ckpt](https://drive.google.com/file/d/1F7PTmvCvr9k6AIbrWEF49LDDMqd94Gda/view?usp=share_link) | [log](https://drive.google.com/file/d/1LvWo2niMWhWAhC8FnU4-GUTKEPY1vRuh/view?usp=share_link) |
| sling_dress           | [pose_resnet_50](/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td_hm_res50_4xb64-210e_deepfashion2_sling_dress_256x192.py) |  256x256   |  0.931  | 0.541 | 24.48 | [ckpt](https://drive.google.com/file/d/1jnKa5VM5pc0pES9ZVM7MSpyWBC10UpRF/view?usp=share_link) | [log](https://drive.google.com/file/d/1lmeIcXJlr4jWOCxW9WJlCxF9Ou-KPNlm/view?usp=share_link) |
