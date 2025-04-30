# Learning delicate local representations for multi-person pose estimation

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58580-8_27">RSN (ECCV'2020)</a></summary>

```bibtex
@misc{cai2020learning,
    title={Learning Delicate Local Representations for Multi-Person Pose Estimation},
    author={Yuanhao Cai and Zhicheng Wang and Zhengxiong Luo and Binyi Yin and Angang Du and Haoqian Wang and Xinyu Zhou and Erjin Zhou and Xiangyu Zhang and Jian Sun},
    year={2020},
    eprint={2003.04030},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

In this paper, we propose a novel method called Residual Steps Network (RSN). RSN aggregates features with the same spatial size (Intra-level features) efficiently to obtain delicate local representations, which retain rich low-level spatial information and result in precise keypoint localization. Additionally, we observe the output features contribute differently to final performance. To tackle this problem, we propose an efficient attention mechanism - Pose Refine Machine (PRM) to make a trade-off between local and global representations in output features and further refine the keypoint locations. Our approach won the 1st place of COCO Keypoint Challenge 2019 and achieves state-of-the-art results on both COCO and MPII benchmarks, without using extra training data and pretrained model. Our single model achieves 78.6 on COCO test-dev, 93.0 on MPII test dataset. Ensembled models achieve 79.2 on COCO test-dev, 77.1 on COCO test-challenge dataset.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522226-5041d16e-41cb-4d31-ae53-dbe21314a697.png">
</div>
