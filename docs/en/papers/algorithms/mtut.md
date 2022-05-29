# Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition with Multimodal Training

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_Hand-Gesture_Recognition_With_Multimodal_CVPR_2019_paper.html">MTUT (CVPR'2019)</a></summary>

```bibtex
@InProceedings{Abavisani_2019_CVPR,
author = {Abavisani, Mahdi and Joze, Hamid Reza Vaezi and Patel, Vishal M.},
title = {Improving the Performance of Unimodal Dynamic Hand-Gesture Recognition With Multimodal Training},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

We present an efficient approach for leveraging the knowledge from multiple modalities in training unimodal 3D convolutional neural networks (3D-CNNs) for the task of dynamic hand gesture recognition. Instead of explicitly combining multimodal information, which is commonplace in many state-of-the-art methods, we propose a different framework in which we embed the knowledge of multiple modalities in individual networks so that each unimodal network can achieve an improved performance. In particular, we dedicate separate networks per available modality and enforce them to collaborate and learn to develop networks with common semantics and better representations. We introduce a "spatiotemporal semantic alignment" loss (SSA) to align the content of the features from different networks. In addition, we regularize this loss with our proposed "focal regularization parameter" to avoid negative knowledge transfer. Experimental results show that our framework improves the test time recognition accuracy of unimodal networks, and provides the state-of-the-art performance on various dynamic hand gesture recognition datasets.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/26127467/170655378-e0db31cc-f9c3-43c3-909a-13ed871b290a.png">
</div>
