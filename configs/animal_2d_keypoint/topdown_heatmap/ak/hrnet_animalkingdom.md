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
<summary align="right"><a href="https://sutdcv.github.io/Animal-Kingdom/">AnimalKingdom (CVPR'2022)</a></summary>

```bibtex
@InProceedings{
    Ng_2022_CVPR,
    author    = {Ng, Xun Long and Ong, Kian Eng and Zheng, Qichen and Ni, Yun and Yeo, Si Yong and Liu, Jun},
    title     = {Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19023-19034}
 }
```

</details>

Results on AnimalKingdom validation set

| Arch                                                                                                                                      | Input Size | PCK(0.05) | PCK(0.05) paper | ckpt | log |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------- | --------------- | ---- | --- |
| [P1_hrnet_w32](configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb64-210e_animalkingdom_P1-256x256.py)                      | 256x256    | 0.6272    | 0.6342          | ckpt | log |
| [P2_hrnet_w32](configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb64-210e_animalkingdom_P2-256x256.py)                      | 256x256    | 0.3774    | 0.3726          | ckpt | log |
| [P3_mammals_hrnet_w32](configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb64-210e_animalkingdom_P3_mammal-256x256.py)       | 256x256    | 0.5756    | 0.5719          | ckpt | log |
| [P3_amphibians_hrnet_w32](configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb64-210e_animalkingdom_P3_amphibian-256x256.py) | 256x256    | 0.5356    | 0.5432          | ckpt | log |
| [P3_reptiles_hrnet_w32](configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb64-210e_animalkingdom_P3_reptile-256x256.py)     | 256x256    | 0.5       | 0.5             | ckpt | log |
| [P3_birds_hrnet_w32](configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb64-210e_animalkingdom_P3_bird-256x256.py)           | 256x256    | 0.7679    | 0.7636          | ckpt | log |
| [P3_fishes_hrnet_w32](configs/animal_2d_keypoint/topdown_heatmap/ak/td-hm_hrnet-w32_8xb64-210e_animalkingdom_P3_fish-256x256.py)          | 256x256    | 0.643     | 0.636           | ckpt | log |
