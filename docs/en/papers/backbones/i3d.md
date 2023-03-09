# Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html">I3D (CVPR'2017)</a></summary>

```bibtex
@InProceedings{Carreira_2017_CVPR,
  author = {Carreira, Joao and Zisserman, Andrew},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

The paucity of videos in current action classification datasets (UCF-101 and HMDB-51) has made it difficult to identify good video architectures, as most methods obtain similar performance on existing small-scale benchmarks. This paper re-evaluates state-of-the-art architectures in light of the new Kinetics Human Action Video dataset. Kinetics has two orders of magnitude more data, with 400 human action classes and over 400 clips per class, and is collected from realistic, challenging YouTube videos. We provide an analysis on how current architectures fare on the task of action classification on this dataset and how much performance improves on the smaller benchmark datasets after pre-training on Kinetics. We also introduce a new Two-Stream Inflated 3D ConvNet (I3D) that is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to learn seamless spatio-temporal feature extractors from video while leveraging successful ImageNet architecture designs and even their parameters. We show that, after pre-training on Kinetics, I3D models considerably improve upon the state-of-the-art in action classification, reaching 80.2% on HMDB-51 and 97.9% on UCF-101.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/26127467/170657728-ded63bd5-a695-4678-92f8-6a2bd1df0164.png">
</div>
