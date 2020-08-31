## Changelog

 ### v0.6.0 (31/8/2020)

 **Highlights**

1. Add more popular backbones & enrich the [modelzoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html).
- ResNext
- SEResNet
- ResNetV1D
- MobileNetv2
- ShuffleNetv1
- CPM (Convolutional Pose Machine)

2. Add more popular datasets:
- [AIChallenger](https://arxiv.org/abs/1711.06475?context=cs.CV)
- [MPII](http://human-pose.mpi-inf.mpg.de/)
- [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)
- [OCHuman](http://www.liruilong.cn/projects/pose2seg/index.html)

3. Support 2d hand keypoint estimation.
- [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)

4. Support bottom-up inference.


 **New Features**
- Support [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html) dataset ([#52](https://github.com/open-mmlab/mmpose/pull/52)).
- Support [MPII](http://human-pose.mpi-inf.mpg.de/) dataset ([#55](https://github.com/open-mmlab/mmpose/pull/55)).
- Support [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) dataset ([#19](https://github.com/open-mmlab/mmpose/pull/19), [#47](https://github.com/open-mmlab/mmpose/pull/47), [#48](https://github.com/open-mmlab/mmpose/pull/48)).
- Support [OCHuman](http://www.liruilong.cn/projects/pose2seg/index.html) dataset ([#70](https://github.com/open-mmlab/mmpose/pull/70).
- Support [AIChallenger](https://arxiv.org/abs/1711.06475?context=cs.CV) dataset ([#87](https://github.com/open-mmlab/mmpose/pull/87)).
- Support multiple backbones ([#26](https://github.com/open-mmlab/mmpose/pull/26)).
- Support CPM model ([#56](https://github.com/open-mmlab/mmpose/pull/56)).

 **Bug Fixes**
- Fix configs for MPII & MPII-TRB datasets ([#93](https://github.com/open-mmlab/mmpose/pull/93)).
- Fix the bug of missing `test_pipeline` in configs ([#14](https://github.com/open-mmlab/mmpose/pull/14))
- Fix typos ([#27](https://github.com/open-mmlab/mmpose/pull/27), [#28](https://github.com/open-mmlab/mmpose/pull/28), [#50](https://github.com/open-mmlab/mmpose/pull/50), [#53](https://github.com/open-mmlab/mmpose/pull/53), [#63](https://github.com/open-mmlab/mmpose/pull/63))

 **Improvements**
- Update benchmark ([#93](https://github.com/open-mmlab/mmpose/pull/93)).
- Add Dockerfile ([#44](https://github.com/open-mmlab/mmpose/pull/44)).
- Improve unittest coverage and minor fix ([#18](https://github.com/open-mmlab/mmpose/pull/18))
- Support CPUs for train/val/demo ([#34](https://github.com/open-mmlab/mmpose/pull/34)).
- Support bottom-up demo ([#69](https://github.com/open-mmlab/mmpose/pull/69)).
- Add tools to publish model ([#62](https://github.com/open-mmlab/mmpose/pull/62)).
- Enrich the modelzoo ([#64](https://github.com/open-mmlab/mmpose/pull/64), [#68](https://github.com/open-mmlab/mmpose/pull/68), [#82](https://github.com/open-mmlab/mmpose/pull/82)).

 ### v0.5.0 (21/7/2020)

 **Highlights**
- MMPose is released.

 **Main Features**
- Support both top-down and bottom-up pose estimation approaches.
- Achieve higher training efficiency and higher accuracy than other popular codebases (e.g. AlphaPose, HRNet).
- Support various backbone models: ResNet, HRNet, SCNet, Houglass and HigherHRNet.
