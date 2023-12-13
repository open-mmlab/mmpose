# RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/ad94c097-7d51-4b91-b885-d8605e22a0e6" height="360px" alt><br>

RTMO is a one-stage pose estimation model that achieves comparable performance with RTMPose. It has following advantages:

- RTMO is **faster when there are multiple persons** in the image (more than 4).
- RTMO **does not depend on auxiliary human detector** and thus is easier to use.

📌 TRY RTMO NOW

```bash
python demo/inferencer_demo.py $IMAGE --pose2d rtmo --vis-out-dir vis_results
```

## Introduction

Real-time multi-person pose estimation presents significant challenges in balancing speed and precision. While two-stage top-down methods slow down as the number of people in the image increases, existing one-stage methods often fail to simultaneously deliver high accuracy and real-time performance. This paper introduces RTMO, a one-stage pose estimation framework that seamlessly integrates coordinate classification by representing keypoints using dual 1-D heatmaps within the YOLO architecture, achieving accuracy comparable to top-down methods while maintaining high speed. We propose a dynamic coordinate classifier and a tailored loss function for heatmap learning, specifically designed to address the incompatibilities between coordinate classification and dense prediction models. RTMO outperforms state-of-the-art one-stage pose estimators, achieving 1.1% higher AP on COCO while operating about 9 times faster with the same backbone. Our largest model, RTMO-l, attains 74.8% AP on COCO val2017 and 141 FPS on a single V100 GPU, demonstrating its efficiency and accuracy.

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/6a520cdc-516a-4ab2-b503-22952056b55f" width="360px" alt><br>

## Train and Evaluation

Coming Soon.

## Deploy

Coming Soon.

## Citation

If this project benefits your work, please kindly consider citing the original papers:

```bibtex
@misc{lu2023rtmo,
      title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
      author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
      year={2023},
      eprint={2312.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
