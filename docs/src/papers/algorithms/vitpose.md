# Deep high-resolution representation learning for human pose estimation

<!-- [ALGORITHM] -->

<details>

<summary  align="right"><a  href="https://arxiv.org/abs/2204.12484">ViTPose</a></summary>

```bibtex

@misc{https://doi.org/10.48550/arxiv.2204.12484,
  doi = {10.48550/ARXIV.2204.12484},
  url = {https://arxiv.org/abs/2204.12484},
  author = {Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

</details>

## Abstract

<!-- [ABSTRACT] -->

Although no specific domain knowledge is considered in the design, plain vision transformers have shown excellent performance in visual recognition tasks. However, little effort has been made to reveal the potential of such simple structures for pose estimation tasks. In this paper, we show the surprisingly good capabilities of plain vision transformers for pose estimation from various aspects, namely simplicity in model structure, scalability in model size, flexibility in training paradigm, and transferability of knowledge between models, through a simple baseline model called ViTPose. Specifically, ViTPose employs plain and non-hierarchical vision transformers as backbones to extract features for a given person instance and a lightweight decoder for pose estimation. It can be scaled up from 100M to 1B parameters by taking the advantages of the scalable model capacity and high parallelism of transformers, setting a new Pareto front between throughput and performance. Besides, ViTPose is very flexible regarding the attention type, input resolution, pre-training and finetuning strategy, as well as dealing with multiple pose tasks. We also empirically demonstrate that the knowledge of large ViTPose models can be easily transferred to small ones via a simple knowledge token. Experimental results show that our basic ViTPose model outperforms representative methods on the challenging MS COCO Keypoint Detection benchmark, while the largest model sets a new state-of-the-art, i.e., 80.9 AP on the MS COCO test-dev set.
