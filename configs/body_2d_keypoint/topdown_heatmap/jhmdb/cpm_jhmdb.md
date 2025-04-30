<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.html">CPM (CVPR'2016)</a></summary>

```bibtex
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Jhuang_Towards_Understanding_Action_2013_ICCV_paper.html">JHMDB (ICCV'2013)</a></summary>

```bibtex
@inproceedings{Jhuang:ICCV:2013,
  title = {Towards understanding action recognition},
  author = {H. Jhuang and J. Gall and S. Zuffi and C. Schmid and M. J. Black},
  booktitle = {International Conf. on Computer Vision (ICCV)},
  month = Dec,
  pages = {3192-3199},
  year = {2013}
}
```

</details>

Results on Sub-JHMDB dataset

The models are pre-trained on MPII dataset only. NO test-time augmentation (multi-scale /rotation testing) is used.

- Normalized by Person Size

| Split   |                        Arch                        | Input Size | Head | Sho  | Elb  | Wri  | Hip  | Knee | Ank  | Mean |                        ckpt                         |                        log                         |
| :------ | :------------------------------------------------: | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-------------------------------------------------: | :------------------------------------------------: |
| Sub1    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/td-hm_cpm_8xb32-40e_jhmdb-sub1-368x368.py) |  368x368   | 96.1 | 91.9 | 81.0 | 78.9 | 96.6 | 90.8 | 87.3 | 89.5 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368-2d2585c9_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368_20201122.log.json) |
| Sub2    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/td-hm_cpm_8xb32-40e_jhmdb-sub2-368x368.py) |  368x368   | 98.1 | 93.6 | 77.1 | 70.9 | 94.0 | 89.1 | 84.7 | 87.4 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368-fc742f1f_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368_20201122.log.json) |
| Sub3    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/td-hm_cpm_8xb32-40e_jhmdb-sub3-368x368.py) |  368x368   | 97.9 | 94.9 | 87.3 | 84.0 | 98.6 | 94.4 | 86.2 | 92.4 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368-49337155_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368_20201122.log.json) |
| Average |                        cpm                         |  368x368   | 97.4 | 93.5 | 81.5 | 77.9 | 96.4 | 91.4 | 86.1 | 89.8 |                          -                          |                         -                          |

- Normalized by Torso Size

| Split   |                        Arch                        | Input Size | Head | Sho  | Elb  | Wri  | Hip  | Knee | Ank  | Mean |                        ckpt                         |                        log                         |
| :------ | :------------------------------------------------: | :--------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-------------------------------------------------: | :------------------------------------------------: |
| Sub1    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/td-hm_cpm_8xb32-40e_jhmdb-sub1-368x368.py) |  368x368   | 89.0 | 63.0 | 54.0 | 54.9 | 68.2 | 63.1 | 61.2 | 66.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368-2d2585c9_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368_20201122.log.json) |
| Sub2    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/td-hm_cpm_8xb32-40e_jhmdb-sub2-368x368.py) |  368x368   | 90.3 | 57.9 | 46.8 | 44.3 | 60.8 | 58.2 | 62.4 | 61.1 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368-fc742f1f_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368_20201122.log.json) |
| Sub3    | [cpm](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/jhmdb/td-hm_cpm_8xb32-40e_jhmdb-sub3-368x368.py) |  368x368   | 91.0 | 72.6 | 59.9 | 54.0 | 73.2 | 68.5 | 65.8 | 70.3 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368-49337155_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368_20201122.log.json) |
| Average |                        cpm                         |  368x368   | 90.1 | 64.5 | 53.6 | 51.1 | 67.4 | 63.3 | 63.1 | 65.7 |                          -                          |                         -                          |
