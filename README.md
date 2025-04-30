<div align="center">
  <img src="resources/mmpose-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b>OpenMMLab website</b>
    <sup>
      <a href="https://openmmlab.com">
        <i>HOT</i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b>OpenMMLab platform</b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i>TRY IT OUT</i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/merge_stage_test/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/latest/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://img.shields.io/pypi/v/mmpose)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/main/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_demo.svg)](https://openxlab.org.cn/apps?search=mmpose)

[📘Documentation](https://mmpose.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmpose.readthedocs.io/en/latest/installation.html) |
[👀Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html) |
[📜Papers](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html) |
[🆕Update News](https://mmpose.readthedocs.io/en/latest/notes/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmpose/issues/new/choose) |
[🔥RTMPose](/projects/rtmpose/)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1072798105428299817" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

English | [简体中文](README_CN.md)

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The main branch works with **PyTorch 1.8+**.

https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4

<br/>

<details close>
<summary><b>Major Features</b></summary>

- **Support diverse tasks**

  We support a wide spectrum of mainstream pose analysis tasks in current research community, including 2d multi-person human pose estimation, 2d hand pose estimation, 2d face landmark detection, 133 keypoint whole-body human pose estimation, 3d human mesh recovery, fashion landmark detection and animal pose estimation.
  See [Demo](demo/docs/en) for more information.

- **Higher efficiency and higher accuracy**

  MMPose implements multiple state-of-the-art (SOTA) deep learning models, including both top-down & bottom-up approaches. We achieve faster training speed and higher accuracy than other popular codebases, such as [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
  See [benchmark.md](docs/en/notes/benchmark.md) for more information.

- **Support for various datasets**

  The toolbox directly supports multiple popular and representative datasets, COCO, AIC, MPII, MPII-TRB, OCHuman etc.
  See [dataset_zoo](docs/en/dataset_zoo) for more information.

- **Well designed, tested and documented**

  We decompose MMPose into different components and one can easily construct a customized
  pose estimation framework by combining different modules.
  We provide detailed documentation and API reference, as well as unittests.

</details>

## What's New

- Release [RTMW3D](/projects/rtmpose3d), a real-time model for 3D wholebody pose estimation.

- Release [RTMO](/projects/rtmo), a state-of-the-art real-time method for multi-person pose estimation.

  ![rtmo](https://github.com/open-mmlab/mmpose/assets/26127467/54d5555a-23e5-4308-89d1-f0c82a6734c2)

- Release [RTMW](/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md) models in various sizes ranging from RTMW-m to RTMW-x. The input sizes include `256x192` and `384x288`. This provides flexibility to select the right model for different speed and accuracy requirements.

- Support inference of [PoseAnything](/projects/pose_anything). Web demo is available [here](https://openxlab.org.cn/apps/detail/orhir/Pose-Anything).

- Support for new datasets:

  - (ICCV 2015) [300VW](/docs/en/dataset_zoo/2d_face_keypoint.md)

- Welcome to use the [*MMPose project*](/projects/README.md). Here, you can discover the latest features and algorithms in MMPose and quickly share your ideas and code implementations with the community. Adding new features to MMPose has become smoother:

  - Provides a simple and fast way to add new algorithms, features, and applications to MMPose.
  - More flexible code structure and style, fewer restrictions, and a shorter code review process.
  - Utilize the powerful capabilities of MMPose in the form of independent projects without being constrained by the code framework.
  - Newly added projects include:
    - [RTMPose](/projects/rtmpose/)
    - [RTMO](/projects/rtmo/)
    - [RTMPose3D](/projects/rtmpose3d/)
    - [PoseAnything](/projects/pose_anything/)
    - [YOLOX-Pose](/projects/yolox_pose/)
    - [MMPose4AIGC](/projects/mmpose4aigc/)
    - [Simple Keypoints](/projects/skps/)
    - [Just Dance](/projects/just_dance/)
    - [Uniformer](/projects/uniformer/)
  - Start your journey as an MMPose contributor with a simple [example project](/projects/example_project/), and let's build a better MMPose together!

<br/>

- January 4, 2024: MMPose [v1.3.0](https://github.com/open-mmlab/mmpose/releases/tag/v1.3.0) has been officially released, with major updates including:

  - Support for new datasets: ExLPose, H3WB
  - Release of new RTMPose series models: RTMO, RTMW
  - Support for new algorithm PoseAnything
  - Enhanced Inferencer with optional progress bar and improved affinity for one-stage methods

  Please check the complete [release notes](https://github.com/open-mmlab/mmpose/releases/tag/v1.3.0) for more details on the updates brought by MMPose v1.3.0!

## 0.x / 1.x Migration

MMPose v1.0.0 is a major update, including many API and config file changes. Currently, a part of the algorithms have been migrated to v1.0.0, and the remaining algorithms will be completed in subsequent versions. We will show the migration progress in this [Roadmap](https://github.com/open-mmlab/mmpose/issues/2258).

If your algorithm has not been migrated, you can continue to use the [0.x branch](https://github.com/open-mmlab/mmpose/tree/0.x) and [old documentation](https://mmpose.readthedocs.io/en/0.x/).

## Installation

Please refer to [installation.md](https://mmpose.readthedocs.io/en/latest/installation.html) for more detailed installation and dataset preparation.

## Getting Started

We provided a series of tutorials about the basic usage of MMPose for new users:

1. For the basic usage of MMPose:

   - [A 20-minute Tour to MMPose](https://mmpose.readthedocs.io/en/latest/guide_to_framework.html)
   - [Demos](https://mmpose.readthedocs.io/en/latest/demos.html)
   - [Inference](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html)
   - [Configs](https://mmpose.readthedocs.io/en/latest/user_guides/configs.html)
   - [Prepare Datasets](https://mmpose.readthedocs.io/en/latest/user_guides/prepare_datasets.html)
   - [Train and Test](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html)
   - [Deployment](https://mmpose.readthedocs.io/en/latest/user_guides/how_to_deploy.html)
   - [Model Analysis](https://mmpose.readthedocs.io/en/latest/user_guides/model_analysis.html)
   - [Dataset Annotation and Preprocessing](https://mmpose.readthedocs.io/en/latest/user_guides/dataset_tools.html)

2. For developers who wish to develop based on MMPose:

   - [Learn about Codecs](https://mmpose.readthedocs.io/en/latest/advanced_guides/codecs.html)
   - [Dataflow in MMPose](https://mmpose.readthedocs.io/en/latest/advanced_guides/dataflow.html)
   - [Implement New Models](https://mmpose.readthedocs.io/en/latest/advanced_guides/implement_new_models.html)
   - [Customize Datasets](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html)
   - [Customize Data Transforms](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_transforms.html)
   - [Customize Evaluation](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_evaluation.html)
   - [Customize Optimizer](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_optimizer.html)
   - [Customize Logging](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_logging.html)
   - [How to Deploy](https://mmpose.readthedocs.io/en/latest/user_guides/how_to_deploy.html)
   - [Model Analysis](https://mmpose.readthedocs.io/en/latest/user_guides/model_analysis.html)
   - [Migration Guide](https://mmpose.readthedocs.io/en/latest/migration.html)

3. For researchers and developers who are willing to contribute to MMPose:

   - [Contribution Guide](https://mmpose.readthedocs.io/en/latest/contribution_guide.html)

4. For some common issues, we provide a FAQ list:

   - [FAQ](https://mmpose.readthedocs.io/en/latest/faq.html)

## Model Zoo

Results and models are available in the **README.md** of each method's config directory.
A summary can be found in the [Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html) page.

<details open>
<summary><b>Supported algorithms:</b></summary>

- [x] [DeepPose](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#deeppose-cvpr-2014) (CVPR'2014)
- [x] [CPM](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#cpm-cvpr-2016) (CVPR'2016)
- [x] [Hourglass](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hourglass-eccv-2016) (ECCV'2016)
- [x] [SimpleBaseline3D](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#simplebaseline3d-iccv-2017) (ICCV'2017)
- [ ] [Associative Embedding](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#associative-embedding-nips-2017) (NeurIPS'2017)
- [x] [SimpleBaseline2D](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#simplebaseline2d-eccv-2018) (ECCV'2018)
- [x] [DSNT](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#dsnt-2018) (ArXiv'2021)
- [x] [HRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrnet-cvpr-2019) (CVPR'2019)
- [x] [IPR](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#ipr-eccv-2018) (ECCV'2018)
- [x] [VideoPose3D](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#videopose3d-cvpr-2019) (CVPR'2019)
- [x] [HRNetv2](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrnetv2-tpami-2019) (TPAMI'2019)
- [x] [MSPN](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#mspn-arxiv-2019) (ArXiv'2019)
- [x] [SCNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#scnet-cvpr-2020) (CVPR'2020)
- [ ] [HigherHRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#higherhrnet-cvpr-2020) (CVPR'2020)
- [x] [RSN](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#rsn-eccv-2020) (ECCV'2020)
- [x] [InterNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#internet-eccv-2020) (ECCV'2020)
- [ ] [VoxelPose](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#voxelpose-eccv-2020) (ECCV'2020)
- [x] [LiteHRNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#litehrnet-cvpr-2021) (CVPR'2021)
- [x] [ViPNAS](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#vipnas-cvpr-2021) (CVPR'2021)
- [x] [Debias-IPR](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#debias-ipr-iccv-2021) (ICCV'2021)
- [x] [SimCC](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#simcc-eccv-2022) (ECCV'2022)

</details>

<details open>
<summary><b>Supported techniques:</b></summary>

- [x] [FPN](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#fpn-cvpr-2017) (CVPR'2017)
- [x] [FP16](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#fp16-arxiv-2017) (ArXiv'2017)
- [x] [Wingloss](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#wingloss-cvpr-2018) (CVPR'2018)
- [x] [AdaptiveWingloss](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#adaptivewingloss-iccv-2019) (ICCV'2019)
- [x] [DarkPose](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#darkpose-cvpr-2020) (CVPR'2020)
- [x] [UDP](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#udp-cvpr-2020) (CVPR'2020)
- [x] [Albumentations](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#albumentations-information-2020) (Information'2020)
- [x] [SoftWingloss](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#softwingloss-tip-2021) (TIP'2021)
- [x] [RLE](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#rle-iccv-2021) (ICCV'2021)

</details>

<details open>
<summary><b>Supported datasets:</b></summary>

- [x] [AFLW](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#aflw-iccvw-2011) \[[homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)\] (ICCVW'2011)
- [x] [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#jhmdb-iccv-2013) \[[homepage](http://jhmdb.is.tue.mpg.de/dataset)\] (ICCV'2013)
- [x] [COFW](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#cofw-iccv-2013) \[[homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/)\] (ICCV'2013)
- [x] [MPII](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#mpii-cvpr-2014) \[[homepage](http://human-pose.mpi-inf.mpg.de/)\] (CVPR'2014)
- [x] [Human3.6M](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#human3-6m-tpami-2014) \[[homepage](http://vision.imar.ro/human3.6m/description.php)\] (TPAMI'2014)
- [x] [COCO](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#coco-eccv-2014) \[[homepage](http://cocodataset.org/)\] (ECCV'2014)
- [x] [CMU Panoptic](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#cmu-panoptic-iccv-2015) \[[homepage](http://domedb.perception.cs.cmu.edu/)\] (ICCV'2015)
- [x] [300VW](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#300w-imavis-2016) \[[homepage](https://ibug.doc.ic.ac.uk/resources/300-VW/)\] (ICCV'2015)
- [x] [DeepFashion](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#deepfashion-cvpr-2016) \[[homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)\] (CVPR'2016)
- [x] [300W](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#300w-imavis-2016) \[[homepage](https://ibug.doc.ic.ac.uk/resources/300-W/)\] (IMAVIS'2016)
- [x] [RHD](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#rhd-iccv-2017) \[[homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)\] (ICCV'2017)
- [x] [CMU Panoptic HandDB](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#cmu-panoptic-handdb-cvpr-2017) \[[homepage](http://domedb.perception.cs.cmu.edu/handdb.html)\] (CVPR'2017)
- [x] [AI Challenger](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#ai-challenger-arxiv-2017) \[[homepage](https://github.com/AIChallenger/AI_Challenger_2017)\] (ArXiv'2017)
- [x] [MHP](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#mhp-acm-mm-2018) \[[homepage](https://lv-mhp.github.io/dataset)\] (ACM MM'2018)
- [x] [WFLW](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#wflw-cvpr-2018) \[[homepage](https://wywu.github.io/projects/LAB/WFLW.html)\] (CVPR'2018)
- [x] [PoseTrack18](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#posetrack18-cvpr-2018) \[[homepage](https://posetrack.net/users/download.php)\] (CVPR'2018)
- [x] [OCHuman](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#ochuman-cvpr-2019) \[[homepage](https://github.com/liruilong940607/OCHumanApi)\] (CVPR'2019)
- [x] [CrowdPose](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#crowdpose-cvpr-2019) \[[homepage](https://github.com/Jeff-sjtu/CrowdPose)\] (CVPR'2019)
- [x] [MPII-TRB](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#mpii-trb-iccv-2019) \[[homepage](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)\] (ICCV'2019)
- [x] [FreiHand](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#freihand-iccv-2019) \[[homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/)\] (ICCV'2019)
- [x] [Animal-Pose](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#animal-pose-iccv-2019) \[[homepage](https://sites.google.com/view/animal-pose/)\] (ICCV'2019)
- [x] [OneHand10K](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#onehand10k-tcsvt-2019) \[[homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)\] (TCSVT'2019)
- [x] [Vinegar Fly](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#vinegar-fly-nature-methods-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Nature Methods'2019)
- [x] [Desert Locust](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#desert-locust-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [Grévy’s Zebra](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#grevys-zebra-elife-2019) \[[homepage](https://github.com/jgraving/DeepPoseKit-Data)\] (Elife'2019)
- [x] [ATRW](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#atrw-acm-mm-2020) \[[homepage](https://cvwc2019.github.io/challenge.html)\] (ACM MM'2020)
- [x] [Halpe](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#halpe-cvpr-2020) \[[homepage](https://github.com/Fang-Haoshu/Halpe-FullBody/)\] (CVPR'2020)
- [x] [COCO-WholeBody](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#coco-wholebody-eccv-2020) \[[homepage](https://github.com/jin-s13/COCO-WholeBody/)\] (ECCV'2020)
- [x] [MacaquePose](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#macaquepose-biorxiv-2020) \[[homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)\] (bioRxiv'2020)
- [x] [InterHand2.6M](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#interhand2-6m-eccv-2020) \[[homepage](https://mks0601.github.io/InterHand2.6M/)\] (ECCV'2020)
- [x] [AP-10K](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#ap-10k-neurips-2021) \[[homepage](https://github.com/AlexTheBad/AP-10K)\] (NeurIPS'2021)
- [x] [Horse-10](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#horse-10-wacv-2021) \[[homepage](http://www.mackenziemathislab.org/horse10)\] (WACV'2021)
- [x] [Human-Art](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#human-art-cvpr-2023) \[[homepage](https://idea-research.github.io/HumanArt/)\] (CVPR'2023)
- [x] [LaPa](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#lapa-aaai-2020) \[[homepage](https://github.com/JDAI-CV/lapa-dataset)\] (AAAI'2020)
- [x] [UBody](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/datasets.html#ubody-cvpr-2023) \[[homepage](https://github.com/IDEA-Research/OSX)\] (CVPR'2023)

</details>

<details open>
<summary><b>Supported backbones:</b></summary>

- [x] [AlexNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#alexnet-neurips-2012) (NeurIPS'2012)
- [x] [VGG](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#vgg-iclr-2015) (ICLR'2015)
- [x] [ResNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#resnet-cvpr-2016) (CVPR'2016)
- [x] [ResNext](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#resnext-cvpr-2017) (CVPR'2017)
- [x] [SEResNet](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#seresnet-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV1](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#shufflenetv1-cvpr-2018) (CVPR'2018)
- [x] [ShufflenetV2](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#shufflenetv2-eccv-2018) (ECCV'2018)
- [x] [MobilenetV2](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#mobilenetv2-cvpr-2018) (CVPR'2018)
- [x] [ResNetV1D](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#resnetv1d-cvpr-2019) (CVPR'2019)
- [x] [ResNeSt](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#resnest-arxiv-2020) (ArXiv'2020)
- [x] [Swin](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#swin-cvpr-2021) (CVPR'2021)
- [x] [HRFormer](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#hrformer-nips-2021) (NIPS'2021)
- [x] [PVT](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#pvt-iccv-2021) (ICCV'2021)
- [x] [PVTV2](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html#pvtv2-cvmj-2022) (CVMJ'2022)

</details>

### Model Request

We will keep up with the latest progress of the community, and support more popular algorithms and frameworks. If you have any feature requests, please feel free to leave a comment in [MMPose Roadmap](https://github.com/open-mmlab/mmpose/issues/2258).

## Contributing

We appreciate all contributions to improve MMPose. Please refer to [CONTRIBUTING.md](https://mmpose.readthedocs.io/en/latest/contribution_guide.html) for the contributing guideline.

## Acknowledgement

MMPose is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
