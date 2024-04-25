# Pose Anything: A Graph-Based Approach for Category-Agnostic Pose Estimation

## [Paper](https://arxiv.org/pdf/2311.17891.pdf) | [Project Page](https://orhir.github.io/pose-anything/) | [Official Repo](https://github.com/orhir/PoseAnything)

By [Or Hirschorn](https://scholar.google.co.il/citations?user=GgFuT_QAAAAJ&hl=iw&oi=ao)
and [Shai Avidan](https://scholar.google.co.il/citations?hl=iw&user=hpItE1QAAAAJ)

![Teaser Figure](https://github.com/open-mmlab/mmpose/assets/26127467/96480360-1a80-41f6-88d3-d6c747506a7e)

## Introduction

We present a novel approach to CAPE that leverages the inherent geometrical
relations between keypoints through a newly designed Graph Transformer Decoder.
By capturing and incorporating this crucial structural information, our method
enhances the accuracy of keypoint localization, marking a significant departure
from conventional CAPE techniques that treat keypoints as isolated entities.

## Citation

If you find this useful, please cite this work as follows:

```bibtex
@misc{hirschorn2023pose,
      title={Pose Anything: A Graph-Based Approach for Category-Agnostic Pose Estimation},
      author={Or Hirschorn and Shai Avidan},
      year={2023},
      eprint={2311.17891},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Getting Started

📣 Pose Anything is available on OpenXLab now. [\[Try it online\]](https://openxlab.org.cn/apps/detail/orhir/Pose-Anything)

### Install Dependencies

We recommend using a virtual environment for running our code.
After installing MMPose, you can install the rest of the dependencies by
running:

```
pip install timm
```

### Pretrained Weights

The full list of pretrained models can be found in
the [Official Repo](https://github.com/orhir/PoseAnything).

## Demo on Custom Images

***A bigger and more accurate version of the model - COMING SOON!***

Download
the [pretrained model](https://drive.google.com/file/d/1RT1Q8AMEa1kj6k9ZqrtWIKyuR4Jn4Pqc/view?usp=drive_link)
and run:

```
python demo.py --support [path_to_support_image] --query [path_to_query_image] --config configs/demo_b.py --checkpoint [path_to_pretrained_ckpt]
```

***Note:*** The demo code supports any config with suitable checkpoint file.
More pre-trained models can be found in the official repo.

## Training and Testing on MP-100 Dataset

**We currently only support demo on custom images through the MMPose repo.**

**For training and testing on the MP-100 dataset, please refer to
the [Official Repo](https://github.com/orhir/PoseAnything).**

## Acknowledgement

Our code is based on code from:

- [CapeFormer](https://github.com/flyinglynx/CapeFormer)

## License

This project is released under the Apache 2.0 license.
