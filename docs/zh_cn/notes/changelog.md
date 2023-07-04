# Changelog

## **v1.0.0rc1 (14/10/2022)**

**Highlights**

- Release RTMPose, a high-performance real-time pose estimation algorithm with cross-platform deployment and inference support. See details at the [project page](/projects/rtmpose/)
- Support several new algorithms: ViTPose (arXiv'2022), CID (CVPR'2022), DEKR (CVPR'2021)
- Add Inferencer, a convenient inference interface that perform pose estimation and visualization on images, videos and webcam streams with only one line of code
- Introduce *Project*, a new form for rapid and easy implementation of new algorithms and features in MMPose, which is more handy for community contributors

**New Features**

- Support RTMPose ([#1971](https://github.com/open-mmlab/mmpose/pull/1971), [#2024](https://github.com/open-mmlab/mmpose/pull/2024), [#2028](https://github.com/open-mmlab/mmpose/pull/2028), [#2030](https://github.com/open-mmlab/mmpose/pull/2030), [#2040](https://github.com/open-mmlab/mmpose/pull/2040), [#2057](https://github.com/open-mmlab/mmpose/pull/2057))
- Support Inferencer ([#1969](https://github.com/open-mmlab/mmpose/pull/1969))
- Support ViTPose ([#1876](https://github.com/open-mmlab/mmpose/pull/1876), [#2056](https://github.com/open-mmlab/mmpose/pull/2056), [#2058](https://github.com/open-mmlab/mmpose/pull/2058)， [#2065](https://github.com/open-mmlab/mmpose/pull/2065))
- Support CID ([#1907](https://github.com/open-mmlab/mmpose/pull/1907))
- Support DEKR ([#1834](https://github.com/open-mmlab/mmpose/pull/1834), [#1901](https://github.com/open-mmlab/mmpose/pull/1901))
- Support training with multiple datasets ([#1767](https://github.com/open-mmlab/mmpose/pull/1767), [#1930](https://github.com/open-mmlab/mmpose/pull/1930), [#1938](https://github.com/open-mmlab/mmpose/pull/1938), [#2025](https://github.com/open-mmlab/mmpose/pull/2025))
- Add *project* to allow rapid and easy implementation of new models and features ([#1914](https://github.com/open-mmlab/mmpose/pull/1914))

**Improvements**

- Improve documentation quality ([#1846](https://github.com/open-mmlab/mmpose/pull/1846), [#1858](https://github.com/open-mmlab/mmpose/pull/1858), [#1872](https://github.com/open-mmlab/mmpose/pull/1872), [#1899](https://github.com/open-mmlab/mmpose/pull/1899), [#1925](https://github.com/open-mmlab/mmpose/pull/1925), [#1945](https://github.com/open-mmlab/mmpose/pull/1945), [#1952](https://github.com/open-mmlab/mmpose/pull/1952), [#1990](https://github.com/open-mmlab/mmpose/pull/1990), [#2023](https://github.com/open-mmlab/mmpose/pull/2023), [#2042](https://github.com/open-mmlab/mmpose/pull/2042))
- Support visualizing keypoint indices ([#2051](https://github.com/open-mmlab/mmpose/pull/2051))
- Support OpenPose style visualization ([#2055](https://github.com/open-mmlab/mmpose/pull/2055))
- Accelerate image transpose in data pipelines with tensor operation ([#1976](https://github.com/open-mmlab/mmpose/pull/1976))
- Support auto-import modules from registry ([#1961](https://github.com/open-mmlab/mmpose/pull/1961))
- Support keypoint partition metric ([#1944](https://github.com/open-mmlab/mmpose/pull/1944))
- Support SimCC 1D-heatmap visualization ([#1912](https://github.com/open-mmlab/mmpose/pull/1912))
- Support saving predictions and data metainfo in demos ([#1814](https://github.com/open-mmlab/mmpose/pull/1814), [#1879](https://github.com/open-mmlab/mmpose/pull/1879))
- Support SimCC with DARK ([#1870](https://github.com/open-mmlab/mmpose/pull/1870))
- Remove Gaussian blur for offset maps in UDP-regress ([#1815](https://github.com/open-mmlab/mmpose/pull/1815))
- Refactor encoding interface of Codec for better extendibility and easier configuration ([#1781](https://github.com/open-mmlab/mmpose/pull/1781))
- Support evaluating CocoMetric without annotation file ([#1722](https://github.com/open-mmlab/mmpose/pull/1722))
- Improve unit tests ([#1765](https://github.com/open-mmlab/mmpose/pull/1765))

**Bug Fixes**

- Fix repeated warnings from different ranks ([#2053](https://github.com/open-mmlab/mmpose/pull/2053))
- Avoid frequent scope switching when using mmdet inference api ([#2039](https://github.com/open-mmlab/mmpose/pull/2039))
- Remove EMA parameters and message hub data when publishing model checkpoints ([#2036](https://github.com/open-mmlab/mmpose/pull/2036))
- Fix metainfo copying in dataset class ([#2017](https://github.com/open-mmlab/mmpose/pull/2017))
- Fix top-down demo bug when there is no object detected ([#2007](https://github.com/open-mmlab/mmpose/pull/2007))
- Fix config errors ([#1882](https://github.com/open-mmlab/mmpose/pull/1882), [#1906](https://github.com/open-mmlab/mmpose/pull/1906), [#1995](https://github.com/open-mmlab/mmpose/pull/1995))
- Fix image demo failure when GUI is unavailable ([#1968](https://github.com/open-mmlab/mmpose/pull/1968))
- Fix bug in AdaptiveWingLoss ([#1953](https://github.com/open-mmlab/mmpose/pull/1953))
- Fix incorrect importing of RepeatDataset which is deprecated ([#1943](https://github.com/open-mmlab/mmpose/pull/1943))
- Fix bug in bottom-up datasets that ignores images without instances ([#1752](https://github.com/open-mmlab/mmpose/pull/1752), [#1936](https://github.com/open-mmlab/mmpose/pull/1936))
- Fix upstream dependency issues ([#1867](https://github.com/open-mmlab/mmpose/pull/1867), [#1921](https://github.com/open-mmlab/mmpose/pull/1921))
- Fix evaluation issues and update results ([#1763](https://github.com/open-mmlab/mmpose/pull/1763), [#1773](https://github.com/open-mmlab/mmpose/pull/1773), [#1780](https://github.com/open-mmlab/mmpose/pull/1780), [#1850](https://github.com/open-mmlab/mmpose/pull/1850), [#1868](https://github.com/open-mmlab/mmpose/pull/1868))
- Fix local registry missing warnings ([#1849](https://github.com/open-mmlab/mmpose/pull/1849))
- Remove deprecated scripts for model deployment ([#1845](https://github.com/open-mmlab/mmpose/pull/1845))
- Fix a bug in input transformation in BaseHead ([#1843](https://github.com/open-mmlab/mmpose/pull/1843))
- Fix an interface mismatch with MMDetection in webcam demo ([#1813](https://github.com/open-mmlab/mmpose/pull/1813))
- Fix a bug in heatmap visualization that causes incorrect scale ([#1800](https://github.com/open-mmlab/mmpose/pull/1800))
- Add model metafiles ([#1768](https://github.com/open-mmlab/mmpose/pull/1768))

## **v1.0.0rc0 (14/10/2022)**

**New Features**

- Support 4 light-weight pose estimation algorithms: [SimCC](https://doi.org/10.48550/arxiv.2107.03332) (ECCV'2022), [Debias-IPR](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_Removing_the_Bias_of_Integral_Pose_Regression_ICCV_2021_paper.pdf) (ICCV'2021), [IPR](https://arxiv.org/abs/1711.08229) (ECCV'2018), and [DSNT](https://arxiv.org/abs/1801.07372v2) (ArXiv'2018) ([#1628](https://github.com/open-mmlab/mmpose/pull/1628))

**Migrations**

- Add Webcam API in MMPose 1.0 ([#1638](https://github.com/open-mmlab/mmpose/pull/1638), [#1662](https://github.com/open-mmlab/mmpose/pull/1662)) @Ben-Louis
- Add codec for Associative Embedding (beta) ([#1603](https://github.com/open-mmlab/mmpose/pull/1603)) @ly015

**Improvements**

- Add a colab tutorial for MMPose 1.0 ([#1660](https://github.com/open-mmlab/mmpose/pull/1660)) @Tau-J
- Add model index in config folder ([#1710](https://github.com/open-mmlab/mmpose/pull/1710), [#1709](https://github.com/open-mmlab/mmpose/pull/1709), [#1627](https://github.com/open-mmlab/mmpose/pull/1627)) @ly015, @Tau-J, @Ben-Louis
- Update and improve documentation ([#1692](https://github.com/open-mmlab/mmpose/pull/1692), [#1656](https://github.com/open-mmlab/mmpose/pull/1656), [#1681](https://github.com/open-mmlab/mmpose/pull/1681), [#1677](https://github.com/open-mmlab/mmpose/pull/1677), [#1664](https://github.com/open-mmlab/mmpose/pull/1664), [#1659](https://github.com/open-mmlab/mmpose/pull/1659)) @Tau-J, @Ben-Louis, @liqikai9
- Improve config structures and formats ([#1651](https://github.com/open-mmlab/mmpose/pull/1651)) @liqikai9

**Bug Fixes**

- Update mmengine version requirements ([#1715](https://github.com/open-mmlab/mmpose/pull/1715)) @Ben-Louis
- Update dependencies of pre-commit hooks ([#1705](https://github.com/open-mmlab/mmpose/pull/1705)) @Ben-Louis
- Fix mmcv version in DockerFile ([#1704](https://github.com/open-mmlab/mmpose/pull/1704))
- Fix a bug in setting dataset metainfo in configs ([#1684](https://github.com/open-mmlab/mmpose/pull/1684)) @ly015
- Fix a bug in UDP training ([#1682](https://github.com/open-mmlab/mmpose/pull/1682)) @liqikai9
- Fix a bug in Dark decoding ([#1676](https://github.com/open-mmlab/mmpose/pull/1676)) @liqikai9
- Fix bugs in visualization ([#1671](https://github.com/open-mmlab/mmpose/pull/1671), [#1668](https://github.com/open-mmlab/mmpose/pull/1668), [#1657](https://github.com/open-mmlab/mmpose/pull/1657)) @liqikai9, @Ben-Louis
- Fix incorrect flops calculation ([#1669](https://github.com/open-mmlab/mmpose/pull/1669)) @liqikai9
- Fix `tensor.tile` compatibility issue for pytorch 1.6 ([#1658](https://github.com/open-mmlab/mmpose/pull/1658)) @ly015
- Fix compatibility with `MultilevelPixelData` ([#1647](https://github.com/open-mmlab/mmpose/pull/1647)) @liqikai9

## **v1.0.0beta (1/09/2022)**

We are excited to announce the release of MMPose 1.0.0beta.
MMPose 1.0.0beta is the first version of MMPose 1.x, a part of the OpenMMLab 2.0 projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMPose 1.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.
It also provide a general semi-supervised object detection framework, and more strong baselines.

**Highlights**

- **New engines**. MMPose 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

- **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMPose 1.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

- **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmpose.readthedocs.io/en/latest/).

**Breaking Changes**

In this release, we made lots of major refactoring and modifications. Please refer to the [migration guide](../migration.md) for details and migration instructions.

## **v0.28.1 (28/07/2022)**

This release is meant to fix the compatibility with the latest mmcv v1.6.1

## **v0.28.0 (06/07/2022)**

**Highlights**

- Support [TCFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zeng_Not_All_Tokens_Are_Equal_Human-Centric_Visual_Analysis_via_Token_CVPR_2022_paper.html) backbone, CVPR'2022 ([#1447](https://github.com/open-mmlab/mmpose/pull/1447), [#1452](https://github.com/open-mmlab/mmpose/pull/1452)) @zengwang430521

- Add [RLE](https://arxiv.org/abs/2107.11291) models on COCO dataset ([#1424](https://github.com/open-mmlab/mmpose/pull/1424)) @Indigo6, @Ben-Louis, @ly015

- Update swin models with better performance ([#1467](https://github.com/open-mmlab/mmpose/pull/1434)) @jin-s13

**New Features**

- Support [TCFormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zeng_Not_All_Tokens_Are_Equal_Human-Centric_Visual_Analysis_via_Token_CVPR_2022_paper.html) backbone, CVPR'2022 ([#1447](https://github.com/open-mmlab/mmpose/pull/1447), [#1452](https://github.com/open-mmlab/mmpose/pull/1452)) @zengwang430521

- Add [RLE](https://arxiv.org/abs/2107.11291) models on COCO dataset ([#1424](https://github.com/open-mmlab/mmpose/pull/1424)) @Indigo6, @Ben-Louis, @ly015

- Support layer decay optimizer constructor and learning rate decay optimizer constructor ([#1423](https://github.com/open-mmlab/mmpose/pull/1423)) @jin-s13

**Improvements**

- Improve documentation quality ([#1416](https://github.com/open-mmlab/mmpose/pull/1416), [#1421](https://github.com/open-mmlab/mmpose/pull/1421), [#1423](https://github.com/open-mmlab/mmpose/pull/1423), [#1426](https://github.com/open-mmlab/mmpose/pull/1426), [#1458](https://github.com/open-mmlab/mmpose/pull/1458), [#1463](https://github.com/open-mmlab/mmpose/pull/1463)) @ly015, @liqikai9

- Support installation by [mim](https://github.com/open-mmlab/mim) ([#1425](https://github.com/open-mmlab/mmpose/pull/1425)) @liqikai9

- Support PAVI logger ([#1434](https://github.com/open-mmlab/mmpose/pull/1434)) @EvelynWang-0423

- Add progress bar for some demos ([#1454](https://github.com/open-mmlab/mmpose/pull/1454)) @liqikai9

- Webcam API supports quick device setting in terminal commands ([#1466](https://github.com/open-mmlab/mmpose/pull/1466)) @ly015

- Update swin models with better performance ([#1467](https://github.com/open-mmlab/mmpose/pull/1434)) @jin-s13

**Bug Fixes**

- Rename `custom_hooks_config` to `custom_hooks` in configs to align with the documentation ([#1427](https://github.com/open-mmlab/mmpose/pull/1427)) @ly015

- Fix deadlock issue in Webcam API ([#1430](https://github.com/open-mmlab/mmpose/pull/1430)) @ly015

- Fix smoother configs in video 3D demo ([#1457](https://github.com/open-mmlab/mmpose/pull/1457)) @ly015

## **v0.27.0 (07/06/2022)**

**Highlights**

- Support hand gesture recognition

  - Try the demo for gesture recognition
  - Learn more about the algorithm, dataset and experiment results

- Major upgrade to the Webcam API

  - Tutorials (EN|zh_CN)
  - [API Reference](https://mmpose.readthedocs.io/en/latest/api.html#mmpose-apis-webcam)
  - Demo

**New Features**

- Support gesture recognition algorithm [MTUT](https://openaccess.thecvf.com/content_CVPR_2019/html/Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_Hand-Gesture_Recognition_With_Multimodal_CVPR_2019_paper.html) CVPR'2019 and dataset [NVGesture](https://openaccess.thecvf.com/content_cvpr_2016/html/Molchanov_Online_Detection_and_CVPR_2016_paper.html) CVPR'2016 ([#1380](https://github.com/open-mmlab/mmpose/pull/1380)) @Ben-Louis

**Improvements**

- Upgrade Webcam API and related documents ([#1393](https://github.com/open-mmlab/mmpose/pull/1393), [#1404](https://github.com/open-mmlab/mmpose/pull/1404), [#1413](https://github.com/open-mmlab/mmpose/pull/1413)) @ly015

- Support exporting COCO inference result without the annotation file ([#1368](https://github.com/open-mmlab/mmpose/pull/1368)) @liqikai9

- Replace markdownlint with mdformat in CI to avoid the dependence on ruby [#1382](https://github.com/open-mmlab/mmpose/pull/1382) @ly015

- Improve documentation quality ([#1385](https://github.com/open-mmlab/mmpose/pull/1385), [#1394](https://github.com/open-mmlab/mmpose/pull/1394), [#1395](https://github.com/open-mmlab/mmpose/pull/1395), [#1408](https://github.com/open-mmlab/mmpose/pull/1408)) @chubei-oppen, @ly015, @liqikai9

**Bug Fixes**

- Fix xywh->xyxy bbox conversion in dataset sanity check ([#1367](https://github.com/open-mmlab/mmpose/pull/1367)) @jin-s13

- Fix a bug in two-stage 3D keypoint demo ([#1373](https://github.com/open-mmlab/mmpose/pull/1373)) @ly015

- Fix out-dated settings in PVT configs ([#1376](https://github.com/open-mmlab/mmpose/pull/1376)) @ly015

- Fix myst settings for document compiling ([#1381](https://github.com/open-mmlab/mmpose/pull/1381)) @ly015

- Fix a bug in bbox transform ([#1384](https://github.com/open-mmlab/mmpose/pull/1384)) @ly015

- Fix inaccurate description of `min_keypoints` in tracking apis ([#1398](https://github.com/open-mmlab/mmpose/pull/1398)) @pallgeuer

- Fix warning with `torch.meshgrid` ([#1402](https://github.com/open-mmlab/mmpose/pull/1402)) @pallgeuer

- Remove redundant transformer modules from `mmpose.datasets.backbones.utils` ([#1405](https://github.com/open-mmlab/mmpose/pull/1405)) @ly015

## **v0.26.0 (05/05/2022)**

**Highlights**

- Support [RLE (Residual Log-likelihood Estimation)](https://arxiv.org/abs/2107.11291), ICCV'2021 ([#1259](https://github.com/open-mmlab/mmpose/pull/1259)) @Indigo6, @ly015

- Support [Swin Transformer](https://arxiv.org/abs/2103.14030), ICCV'2021 ([#1300](https://github.com/open-mmlab/mmpose/pull/1300)) @yumendecc, @ly015

- Support [PVT](https://arxiv.org/abs/2102.12122), ICCV'2021 and [PVTv2](https://arxiv.org/abs/2106.13797), CVMJ'2022 ([#1343](https://github.com/open-mmlab/mmpose/pull/1343)) @zengwang430521

- Speed up inference and reduce CPU usage by optimizing the pre-processing pipeline ([#1320](https://github.com/open-mmlab/mmpose/pull/1320)) @chenxinfeng4, @liqikai9

**New Features**

- Support [RLE (Residual Log-likelihood Estimation)](https://arxiv.org/abs/2107.11291), ICCV'2021 ([#1259](https://github.com/open-mmlab/mmpose/pull/1259)) @Indigo6, @ly015

- Support [Swin Transformer](https://arxiv.org/abs/2103.14030), ICCV'2021 ([#1300](https://github.com/open-mmlab/mmpose/pull/1300)) @yumendecc, @ly015

- Support [PVT](https://arxiv.org/abs/2102.12122), ICCV'2021 and [PVTv2](https://arxiv.org/abs/2106.13797), CVMJ'2022 ([#1343](https://github.com/open-mmlab/mmpose/pull/1343)) @zengwang430521

- Support [FPN](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html), CVPR'2017 ([#1300](https://github.com/open-mmlab/mmpose/pull/1300)) @yumendecc, @ly015

**Improvements**

- Speed up inference and reduce CPU usage by optimizing the pre-processing pipeline ([#1320](https://github.com/open-mmlab/mmpose/pull/1320)) @chenxinfeng4, @liqikai9

- Video demo supports models that requires multi-frame inputs ([#1300](https://github.com/open-mmlab/mmpose/pull/1300)) @liqikai9, @jin-s13

- Update benchmark regression list ([#1328](https://github.com/open-mmlab/mmpose/pull/1328)) @ly015, @liqikai9

- Remove unnecessary warnings in `TopDownPoseTrack18VideoDataset` ([#1335](https://github.com/open-mmlab/mmpose/pull/1335)) @liqikai9

- Improve documentation quality ([#1313](https://github.com/open-mmlab/mmpose/pull/1313), [#1305](https://github.com/open-mmlab/mmpose/pull/1305)) @Ben-Louis, @ly015

- Update deprecating settings in configs ([#1317](https://github.com/open-mmlab/mmpose/pull/1317)) @ly015

**Bug Fixes**

- Fix a bug in human skeleton grouping that may skip the matching process unexpectedly when `ignore_to_much` is True ([#1341](https://github.com/open-mmlab/mmpose/pull/1341)) @daixinghome

- Fix a GPG key error that leads to CI failure ([#1354](https://github.com/open-mmlab/mmpose/pull/1354)) @ly015

- Fix bugs in distributed training script ([#1338](https://github.com/open-mmlab/mmpose/pull/1338), [#1298](https://github.com/open-mmlab/mmpose/pull/1298)) @ly015

- Fix an upstream bug in xtoccotools that causes incorrect AP(M) results ([#1308](https://github.com/open-mmlab/mmpose/pull/1308)) @jin-s13, @ly015

- Fix indentiation errors in the colab tutorial ([#1298](https://github.com/open-mmlab/mmpose/pull/1298)) @YuanZi1501040205

- Fix incompatible model weight initialization with other OpenMMLab codebases ([#1329](https://github.com/open-mmlab/mmpose/pull/1329)) @274869388

- Fix HRNet FP16 checkpoints download URL ([#1309](https://github.com/open-mmlab/mmpose/pull/1309)) @YinAoXiong

- Fix typos in `body3d_two_stage_video_demo.py` ([#1295](https://github.com/open-mmlab/mmpose/pull/1295)) @mucozcan

**Breaking Changes**

- Refactor bbox processing in datasets and pipelines ([#1311](https://github.com/open-mmlab/mmpose/pull/1311)) @ly015, @Ben-Louis

- The bbox format conversion (xywh to center-scale) and random translation are moved from the dataset to the pipeline. The comparison between new and old version is as below:

v0.26.0v0.25.0Dataset
(e.g. [TopDownCOCODataset](https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/top_down/topdown_coco_dataset.py))

... # Data sample only contains bbox rec.append({    'bbox': obj\['clean_bbox\]\[:4\],    ... })

</td>

<td  valign='top'>

... # Convert bbox from xywh to center-scale center, scale = self.\_xywh2cs(\*obj\['clean_bbox'\]\[:4\]) # Data sample contains center and scale rec.append({    'bbox': obj\['clean_bbox\]\[:4\],    'center': center,    'scale': scale,    ... })

</td>

</tr>

<tr>

<th>Pipeline Config

(e.g. [HRNet+COCO](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py))</th>

<td valign='top'>

... train_pipeline = \[    dict(type='LoadImageFromFile'),    # Convert bbox from xywh to center-scale    dict(type='TopDownGetBboxCenterScale', padding=1.25),    # Randomly shift bbox center    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),    ... \]

</td>

<td valign='top'>

... train_pipeline = \[    dict(type='LoadImageFromFile'),    ... \]

</td>

</tr>

<tr>

<th>Advantage</th>

<td valign='top'>

<li>Simpler data sample content</li>

<li>Flexible bbox format conversion and augmentation</li>

<li>Apply bbox random translation every epoch (instead of only applying once at the annotation loading)

</td>

<td valign='top'>-</td>

</tr>

<tr>

<th>BC Breaking</th>

<td valign='top'>The method `_xywh2cs` of dataset base classes (e.g. [Kpt2dSviewRgbImgTopDownDataset](https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py)) will be deprecated in the future. Custom datasets will need modifications to move the bbox format conversion to pipelines.</td>

<td valign='top'>-</td>

</tr>

</tbody>

</table>

## **v0.25.0 (02/04/2022)**

**Highlights**

- Support Shelf and Campus datasets with pre-trained VoxelPose models, ["3D Pictorial Structures for Multiple Human Pose Estimation"](http://campar.in.tum.de/pub/belagiannis2014cvpr/belagiannis2014cvpr.pdf), CVPR'2014 ([#1225](https://github.com/open-mmlab/mmpose/pull/1225)) @liqikai9, @wusize

- Add `Smoother` module for temporal smoothing of the pose estimation with configurable filters ([#1127](https://github.com/open-mmlab/mmpose/pull/1127)) @ailingzengzzz, @ly015

- Support SmoothNet for pose smoothing, ["SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos"](https://arxiv.org/abs/2112.13715), arXiv'2021 ([#1279](https://github.com/open-mmlab/mmpose/pull/1279)) @ailingzengzzz, @ly015

- Add multiview 3D pose estimation demo ([#1270](https://github.com/open-mmlab/mmpose/pull/1270)) @wusize

**New Features**

- Support Shelf and Campus datasets with pre-trained VoxelPose models, ["3D Pictorial Structures for Multiple Human Pose Estimation"](http://campar.in.tum.de/pub/belagiannis2014cvpr/belagiannis2014cvpr.pdf), CVPR'2014 ([#1225](https://github.com/open-mmlab/mmpose/pull/1225)) @liqikai9, @wusize

- Add `Smoother` module for temporal smoothing of the pose estimation with configurable filters ([#1127](https://github.com/open-mmlab/mmpose/pull/1127)) @ailingzengzzz, @ly015

- Support SmoothNet for pose smoothing, ["SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos"](https://arxiv.org/abs/2112.13715), arXiv'2021 ([#1279](https://github.com/open-mmlab/mmpose/pull/1279)) @ailingzengzzz, @ly015

- Add multiview 3D pose estimation demo ([#1270](https://github.com/open-mmlab/mmpose/pull/1270)) @wusize

- Support multi-machine distributed training ([#1248](https://github.com/open-mmlab/mmpose/pull/1248)) @ly015

**Improvements**

- Update HRFormer configs and checkpoints with relative position bias ([#1245](https://github.com/open-mmlab/mmpose/pull/1245)) @zengwang430521

- Support using different random seed for each distributed node ([#1257](https://github.com/open-mmlab/mmpose/pull/1257), [#1229](https://github.com/open-mmlab/mmpose/pull/1229)) @ly015

- Improve documentation quality ([#1275](https://github.com/open-mmlab/mmpose/pull/1275), [#1255](https://github.com/open-mmlab/mmpose/pull/1255), [#1258](https://github.com/open-mmlab/mmpose/pull/1258), [#1249](https://github.com/open-mmlab/mmpose/pull/1249), [#1247](https://github.com/open-mmlab/mmpose/pull/1247), [#1240](https://github.com/open-mmlab/mmpose/pull/1240), [#1235](https://github.com/open-mmlab/mmpose/pull/1235)) @ly015, @jin-s13, @YoniChechik

**Bug Fixes**

- Fix keypoint index in RHD dataset meta information ([#1265](https://github.com/open-mmlab/mmpose/pull/1265)) @liqikai9

- Fix pre-commit hook unexpected behavior on Windows ([#1282](https://github.com/open-mmlab/mmpose/pull/1282)) @liqikai9

- Remove python-dev installation in CI ([#1276](https://github.com/open-mmlab/mmpose/pull/1276)) @ly015

- Unify hyphens in argument names in tools and demos ([#1271](https://github.com/open-mmlab/mmpose/pull/1271)) @ly015

- Fix ambiguous channel size in `channel_shuffle` that may cause exporting failure (#1242) @PINTO0309

- Fix a bug in Webcam API that causes single-class detectors fail ([#1239](https://github.com/open-mmlab/mmpose/pull/1239)) @674106399

- Fix the issue that `custom_hook` can not be set in configs ([#1236](https://github.com/open-mmlab/mmpose/pull/1236)) @bladrome

- Fix incompatible MMCV version in DockerFile ([#raykindle](https://github.com/open-mmlab/mmpose/pull/raykindle))

- Skip invisible joints in visualization ([#1228](https://github.com/open-mmlab/mmpose/pull/1228)) @womeier

## **v0.24.0 (07/03/2022)**

**Highlights**

- Support HRFormer ["HRFormer: High-Resolution Vision Transformer for Dense Predict"](https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html), NeurIPS'2021 ([#1203](https://github.com/open-mmlab/mmpose/pull/1203)) @zengwang430521

- Support Windows installation with pip ([#1213](https://github.com/open-mmlab/mmpose/pull/1213)) @jin-s13, @ly015

- Add WebcamAPI documents ([#1187](https://github.com/open-mmlab/mmpose/pull/1187)) @ly015

**New Features**

- Support HRFormer ["HRFormer: High-Resolution Vision Transformer for Dense Predict"](https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html), NeurIPS'2021 ([#1203](https://github.com/open-mmlab/mmpose/pull/1203)) @zengwang430521

- Support Windows installation with pip ([#1213](https://github.com/open-mmlab/mmpose/pull/1213)) @jin-s13, @ly015

- Support CPU training with mmcv \< v1.4.4 ([#1161](https://github.com/open-mmlab/mmpose/pull/1161)) @EasonQYS, @ly015

- Add "Valentine Magic" demo with WebcamAPI ([#1189](https://github.com/open-mmlab/mmpose/pull/1189), [#1191](https://github.com/open-mmlab/mmpose/pull/1191)) @liqikai9

**Improvements**

- Refactor multi-view 3D pose estimation framework towards better modularization and expansibility ([#1196](https://github.com/open-mmlab/mmpose/pull/1196)) @wusize

- Add WebcamAPI documents and tutorials ([#1187](https://github.com/open-mmlab/mmpose/pull/1187)) @ly015

- Refactor dataset evaluation interface to align with other OpenMMLab codebases ([#1209](https://github.com/open-mmlab/mmpose/pull/1209)) @ly015

- Add deprecation message for deploy tools since [MMDeploy](https://github.com/open-mmlab/mmdeploy) has supported MMPose ([#1207](https://github.com/open-mmlab/mmpose/pull/1207)) @QwQ2000

- Improve documentation quality ([#1206](https://github.com/open-mmlab/mmpose/pull/1206), [#1161](https://github.com/open-mmlab/mmpose/pull/1161)) @ly015

- Switch to OpenMMLab official pre-commit-hook for copyright check ([#1214](https://github.com/open-mmlab/mmpose/pull/1214)) @ly015

**Bug Fixes**

- Fix hard-coded data collating and scattering in inference ([#1175](https://github.com/open-mmlab/mmpose/pull/1175)) @ly015

- Fix model configs on JHMDB dataset ([#1188](https://github.com/open-mmlab/mmpose/pull/1188)) @jin-s13

- Fix area calculation in pose tracking inference ([#1197](https://github.com/open-mmlab/mmpose/pull/1197)) @pallgeuer

- Fix registry scope conflict of module wrapper ([#1204](https://github.com/open-mmlab/mmpose/pull/1204)) @ly015

- Update MMCV installation in CI and documents ([#1205](https://github.com/open-mmlab/mmpose/pull/1205))

- Fix incorrect color channel order in visualization functions ([#1212](https://github.com/open-mmlab/mmpose/pull/1212)) @ly015

## **v0.23.0 (11/02/2022)**

**Highlights**

- Add [MMPose Webcam API](https://github.com/open-mmlab/mmpose/tree/master/tools/webcam): A simple yet powerful tools to develop interactive webcam applications with MMPose functions. ([#1178](https://github.com/open-mmlab/mmpose/pull/1178), [#1173](https://github.com/open-mmlab/mmpose/pull/1173), [#1173](https://github.com/open-mmlab/mmpose/pull/1173), [#1143](https://github.com/open-mmlab/mmpose/pull/1143), [#1094](https://github.com/open-mmlab/mmpose/pull/1094), [#1133](https://github.com/open-mmlab/mmpose/pull/1133), [#1098](https://github.com/open-mmlab/mmpose/pull/1098), [#1160](https://github.com/open-mmlab/mmpose/pull/1160)) @ly015, @jin-s13, @liqikai9, @wusize, @luminxu, @zengwang430521 @mzr1996

**New Features**

- Add [MMPose Webcam API](https://github.com/open-mmlab/mmpose/tree/master/tools/webcam): A simple yet powerful tools to develop interactive webcam applications with MMPose functions. ([#1178](https://github.com/open-mmlab/mmpose/pull/1178), [#1173](https://github.com/open-mmlab/mmpose/pull/1173), [#1173](https://github.com/open-mmlab/mmpose/pull/1173), [#1143](https://github.com/open-mmlab/mmpose/pull/1143), [#1094](https://github.com/open-mmlab/mmpose/pull/1094), [#1133](https://github.com/open-mmlab/mmpose/pull/1133), [#1098](https://github.com/open-mmlab/mmpose/pull/1098), [#1160](https://github.com/open-mmlab/mmpose/pull/1160)) @ly015, @jin-s13, @liqikai9, @wusize, @luminxu, @zengwang430521 @mzr1996

- Support ConcatDataset ([#1139](https://github.com/open-mmlab/mmpose/pull/1139)) @Canwang-sjtu

- Support CPU training and testing ([#1157](https://github.com/open-mmlab/mmpose/pull/1157)) @ly015

**Improvements**

- Add multi-processing configurations to speed up distributed training and testing ([#1146](https://github.com/open-mmlab/mmpose/pull/1146)) @ly015

- Add default runtime config ([#1145](https://github.com/open-mmlab/mmpose/pull/1145))

- Upgrade isort in pre-commit hook ([#1179](https://github.com/open-mmlab/mmpose/pull/1179)) @liqikai9

- Update README and documents ([#1171](https://github.com/open-mmlab/mmpose/pull/1171), [#1167](https://github.com/open-mmlab/mmpose/pull/1167), [#1153](https://github.com/open-mmlab/mmpose/pull/1153), [#1149](https://github.com/open-mmlab/mmpose/pull/1149), [#1148](https://github.com/open-mmlab/mmpose/pull/1148), [#1147](https://github.com/open-mmlab/mmpose/pull/1147), [#1140](https://github.com/open-mmlab/mmpose/pull/1140)) @jin-s13, @wusize, @TommyZihao, @ly015

**Bug Fixes**

- Fix undeterministic behavior in pre-commit hooks ([#1136](https://github.com/open-mmlab/mmpose/pull/1136)) @jin-s13

- Deprecate the support for "python setup.py test" ([#1179](https://github.com/open-mmlab/mmpose/pull/1179)) @ly015

- Fix incompatible settings with MMCV on HSigmoid default parameters ([#1132](https://github.com/open-mmlab/mmpose/pull/1132)) @ly015

- Fix albumentation installation ([#1184](https://github.com/open-mmlab/mmpose/pull/1184)) @BIGWangYuDong

## **v0.22.0 (04/01/2022)**

**Highlights**

- Support VoxelPose ["VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment"](https://arxiv.org/abs/2004.06239), ECCV'2020 ([#1050](https://github.com/open-mmlab/mmpose/pull/1050)) @wusize

- Support Soft Wing loss ["Structure-Coherent Deep Feature Learning for Robust Face Alignment"](https://linchunze.github.io/papers/TIP21_Structure_coherent_FA.pdf), TIP'2021 ([#1077](https://github.com/open-mmlab/mmpose/pull/1077)) @jin-s13

- Support Adaptive Wing loss ["Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"](https://arxiv.org/abs/1904.07399), ICCV'2019 ([#1072](https://github.com/open-mmlab/mmpose/pull/1072)) @jin-s13

**New Features**

- Support VoxelPose ["VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment"](https://arxiv.org/abs/2004.06239), ECCV'2020 ([#1050](https://github.com/open-mmlab/mmpose/pull/1050)) @wusize

- Support Soft Wing loss ["Structure-Coherent Deep Feature Learning for Robust Face Alignment"](https://linchunze.github.io/papers/TIP21_Structure_coherent_FA.pdf), TIP'2021 ([#1077](https://github.com/open-mmlab/mmpose/pull/1077)) @jin-s13

- Support Adaptive Wing loss ["Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"](https://arxiv.org/abs/1904.07399), ICCV'2019 ([#1072](https://github.com/open-mmlab/mmpose/pull/1072)) @jin-s13

- Add LiteHRNet-18 Checkpoints trained on COCO. ([#1120](https://github.com/open-mmlab/mmpose/pull/1120)) @jin-s13

**Improvements**

- Improve documentation quality ([#1115](https://github.com/open-mmlab/mmpose/pull/1115), [#1111](https://github.com/open-mmlab/mmpose/pull/1111), [#1105](https://github.com/open-mmlab/mmpose/pull/1105), [#1087](https://github.com/open-mmlab/mmpose/pull/1087), [#1086](https://github.com/open-mmlab/mmpose/pull/1086), [#1085](https://github.com/open-mmlab/mmpose/pull/1085), [#1084](https://github.com/open-mmlab/mmpose/pull/1084), [#1083](https://github.com/open-mmlab/mmpose/pull/1083), [#1124](https://github.com/open-mmlab/mmpose/pull/1124), [#1070](https://github.com/open-mmlab/mmpose/pull/1070), [#1068](https://github.com/open-mmlab/mmpose/pull/1068)) @jin-s13, @liqikai9, @ly015

- Support CircleCI ([#1074](https://github.com/open-mmlab/mmpose/pull/1074)) @ly015

- Skip unit tests in CI when only document files were changed ([#1074](https://github.com/open-mmlab/mmpose/pull/1074), [#1041](https://github.com/open-mmlab/mmpose/pull/1041)) @QwQ2000, @ly015

- Support file_client_args in LoadImageFromFile ([#1076](https://github.com/open-mmlab/mmpose/pull/1076)) @jin-s13

**Bug Fixes**

- Fix a bug in Dark UDP postprocessing that causes error when the channel number is large. ([#1079](https://github.com/open-mmlab/mmpose/pull/1079), [#1116](https://github.com/open-mmlab/mmpose/pull/1116)) @X00123, @jin-s13

- Fix hard-coded `sigmas` in bottom-up image demo ([#1107](https://github.com/open-mmlab/mmpose/pull/1107), [#1101](https://github.com/open-mmlab/mmpose/pull/1101)) @chenxinfeng4, @liqikai9

- Fix unstable checks in unit tests ([#1112](https://github.com/open-mmlab/mmpose/pull/1112)) @ly015

- Do not destroy NULL windows if `args.show==False` in demo scripts ([#1104](https://github.com/open-mmlab/mmpose/pull/1104)) @bladrome

## **v0.21.0 (06/12/2021)**

**Highlights**

- Support ["Learning Temporal Pose Estimation from Sparsely-Labeled Videos"](https://arxiv.org/abs/1906.04016), NeurIPS'2019 ([#932](https://github.com/open-mmlab/mmpose/pull/932), [#1006](https://github.com/open-mmlab/mmpose/pull/1006), [#1036](https://github.com/open-mmlab/mmpose/pull/1036), [#1060](https://github.com/open-mmlab/mmpose/pull/1060)) @liqikai9

- Add ViPNAS-MobileNetV3 models ([#1025](https://github.com/open-mmlab/mmpose/pull/1025)) @luminxu, @jin-s13

- Add inference speed benchmark ([#1028](https://github.com/open-mmlab/mmpose/pull/1028), [#1034](https://github.com/open-mmlab/mmpose/pull/1034), [#1044](https://github.com/open-mmlab/mmpose/pull/1044)) @liqikai9

**New Features**

- Support ["Learning Temporal Pose Estimation from Sparsely-Labeled Videos"](https://arxiv.org/abs/1906.04016), NeurIPS'2019 ([#932](https://github.com/open-mmlab/mmpose/pull/932), [#1006](https://github.com/open-mmlab/mmpose/pull/1006), [#1036](https://github.com/open-mmlab/mmpose/pull/1036)) @liqikai9

- Add ViPNAS-MobileNetV3 models ([#1025](https://github.com/open-mmlab/mmpose/pull/1025)) @luminxu, @jin-s13

- Add light-weight top-down models for whole-body keypoint detection ([#1009](https://github.com/open-mmlab/mmpose/pull/1009), [#1020](https://github.com/open-mmlab/mmpose/pull/1020), [#1055](https://github.com/open-mmlab/mmpose/pull/1055)) @luminxu, @ly015

- Add HRNet checkpoints with various settings on PoseTrack18 ([#1035](https://github.com/open-mmlab/mmpose/pull/1035)) @liqikai9

**Improvements**

- Add inference speed benchmark ([#1028](https://github.com/open-mmlab/mmpose/pull/1028), [#1034](https://github.com/open-mmlab/mmpose/pull/1034), [#1044](https://github.com/open-mmlab/mmpose/pull/1044)) @liqikai9

- Update model metafile format ([#1001](https://github.com/open-mmlab/mmpose/pull/1001)) @ly015

- Support minus output feature index in mobilenet_v3 ([#1005](https://github.com/open-mmlab/mmpose/pull/1005)) @luminxu

- Improve documentation quality ([#1018](https://github.com/open-mmlab/mmpose/pull/1018), [#1026](https://github.com/open-mmlab/mmpose/pull/1026), [#1027](https://github.com/open-mmlab/mmpose/pull/1027), [#1031](https://github.com/open-mmlab/mmpose/pull/1031), [#1038](https://github.com/open-mmlab/mmpose/pull/1038), [#1046](https://github.com/open-mmlab/mmpose/pull/1046), [#1056](https://github.com/open-mmlab/mmpose/pull/1056), [#1057](https://github.com/open-mmlab/mmpose/pull/1057)) @edybk, @luminxu, @ly015, @jin-s13

- Set default random seed in training initialization ([#1030](https://github.com/open-mmlab/mmpose/pull/1030)) @ly015

- Skip CI when only specific files changed ([#1041](https://github.com/open-mmlab/mmpose/pull/1041), [#1059](https://github.com/open-mmlab/mmpose/pull/1059)) @QwQ2000, @ly015

- Automatically cancel uncompleted action runs when new commit arrives ([#1053](https://github.com/open-mmlab/mmpose/pull/1053)) @ly015

**Bug Fixes**

- Update pose tracking demo to be compatible with latest mmtracking ([#1014](https://github.com/open-mmlab/mmpose/pull/1014)) @jin-s13

- Fix symlink creation failure when installed in Windows environments ([#1039](https://github.com/open-mmlab/mmpose/pull/1039)) @QwQ2000

- Fix AP-10K dataset sigmas ([#1040](https://github.com/open-mmlab/mmpose/pull/1040)) @jin-s13

## **v0.20.0 (01/11/2021)**

**Highlights**

- Add AP-10K dataset for animal pose estimation ([#987](https://github.com/open-mmlab/mmpose/pull/987)) @Annbless, @AlexTheBad, @jin-s13, @ly015

- Support TorchServe ([#979](https://github.com/open-mmlab/mmpose/pull/979)) @ly015

**New Features**

- Add AP-10K dataset for animal pose estimation ([#987](https://github.com/open-mmlab/mmpose/pull/987)) @Annbless, @AlexTheBad, @jin-s13, @ly015

- Add HRNetv2 checkpoints on 300W and COFW datasets ([#980](https://github.com/open-mmlab/mmpose/pull/980)) @jin-s13

- Support TorchServe ([#979](https://github.com/open-mmlab/mmpose/pull/979)) @ly015

**Bug Fixes**

- Fix some deprecated or risky settings in configs ([#963](https://github.com/open-mmlab/mmpose/pull/963), [#976](https://github.com/open-mmlab/mmpose/pull/976), [#992](https://github.com/open-mmlab/mmpose/pull/992)) @jin-s13, @wusize

- Fix issues of default arguments of training and testing scripts ([#970](https://github.com/open-mmlab/mmpose/pull/970), [#985](https://github.com/open-mmlab/mmpose/pull/985)) @liqikai9, @wusize

- Fix heatmap and tag size mismatch in bottom-up with UDP ([#994](https://github.com/open-mmlab/mmpose/pull/994)) @wusize

- Fix python3.9 installation in CI ([#983](https://github.com/open-mmlab/mmpose/pull/983)) @ly015

- Fix model zoo document integrity issue ([#990](https://github.com/open-mmlab/mmpose/pull/990)) @jin-s13

**Improvements**

- Support non-square input shape for bottom-up ([#991](https://github.com/open-mmlab/mmpose/pull/991)) @wusize

- Add image and video resources for demo ([#971](https://github.com/open-mmlab/mmpose/pull/971)) @liqikai9

- Use CUDA docker images to accelerate CI ([#973](https://github.com/open-mmlab/mmpose/pull/973)) @ly015

- Add codespell hook and fix detected typos ([#977](https://github.com/open-mmlab/mmpose/pull/977)) @ly015

## **v0.19.0 (08/10/2021)**

**Highlights**

- Add models for Associative Embedding with Hourglass network backbone ([#906](https://github.com/open-mmlab/mmpose/pull/906), [#955](https://github.com/open-mmlab/mmpose/pull/955)) @jin-s13, @luminxu

- Support COCO-Wholebody-Face and COCO-Wholebody-Hand datasets ([#813](https://github.com/open-mmlab/mmpose/pull/813)) @jin-s13, @innerlee, @luminxu

- Upgrade dataset interface ([#901](https://github.com/open-mmlab/mmpose/pull/901), [#924](https://github.com/open-mmlab/mmpose/pull/924)) @jin-s13, @innerlee, @ly015, @liqikai9

- New style of documentation ([#945](https://github.com/open-mmlab/mmpose/pull/945)) @ly015

**New Features**

- Add models for Associative Embedding with Hourglass network backbone ([#906](https://github.com/open-mmlab/mmpose/pull/906), [#955](https://github.com/open-mmlab/mmpose/pull/955)) @jin-s13, @luminxu

- Support COCO-Wholebody-Face and COCO-Wholebody-Hand datasets ([#813](https://github.com/open-mmlab/mmpose/pull/813)) @jin-s13, @innerlee, @luminxu

- Add pseudo-labeling tool to generate COCO style keypoint annotations with given bounding boxes ([#928](https://github.com/open-mmlab/mmpose/pull/928)) @soltkreig

- New style of documentation ([#945](https://github.com/open-mmlab/mmpose/pull/945)) @ly015

**Bug Fixes**

- Fix segmentation parsing in Macaque dataset preprocessing ([#948](https://github.com/open-mmlab/mmpose/pull/948)) @jin-s13

- Fix dependencies that may lead to CI failure in downstream projects ([#936](https://github.com/open-mmlab/mmpose/pull/936), [#953](https://github.com/open-mmlab/mmpose/pull/953)) @RangiLyu, @ly015

- Fix keypoint order in Human3.6M dataset ([#940](https://github.com/open-mmlab/mmpose/pull/940)) @ttxskk

- Fix unstable image loading for Interhand2.6M ([#913](https://github.com/open-mmlab/mmpose/pull/913)) @zengwang430521

**Improvements**

- Upgrade dataset interface ([#901](https://github.com/open-mmlab/mmpose/pull/901), [#924](https://github.com/open-mmlab/mmpose/pull/924)) @jin-s13, @innerlee, @ly015, @liqikai9

- Improve demo usability and stability ([#908](https://github.com/open-mmlab/mmpose/pull/908), [#934](https://github.com/open-mmlab/mmpose/pull/934)) @ly015

- Standardize model metafile format ([#941](https://github.com/open-mmlab/mmpose/pull/941)) @ly015

- Support `persistent_worker` and several other arguments in configs ([#946](https://github.com/open-mmlab/mmpose/pull/946)) @jin-s13

- Use MMCV root model registry to enable cross-project module building ([#935](https://github.com/open-mmlab/mmpose/pull/935)) @RangiLyu

- Improve the document quality ([#916](https://github.com/open-mmlab/mmpose/pull/916), [#909](https://github.com/open-mmlab/mmpose/pull/909), [#942](https://github.com/open-mmlab/mmpose/pull/942), [#913](https://github.com/open-mmlab/mmpose/pull/913), [#956](https://github.com/open-mmlab/mmpose/pull/956)) @jin-s13, @ly015, @bit-scientist, @zengwang430521

- Improve pull request template ([#952](https://github.com/open-mmlab/mmpose/pull/952), [#954](https://github.com/open-mmlab/mmpose/pull/954)) @ly015

**Breaking Changes**

- Upgrade dataset interface ([#901](https://github.com/open-mmlab/mmpose/pull/901)) @jin-s13, @innerlee, @ly015

## **v0.18.0 (01/09/2021)**

**Bug Fixes**

- Fix redundant model weight loading in pytorch-to-onnx conversion ([#850](https://github.com/open-mmlab/mmpose/pull/850)) @ly015

- Fix a bug in update_model_index.py that may cause pre-commit hook failure([#866](https://github.com/open-mmlab/mmpose/pull/866)) @ly015

- Fix a bug in interhand_3d_head ([#890](https://github.com/open-mmlab/mmpose/pull/890)) @zengwang430521

- Fix pose tracking demo failure caused by out-of-date configs ([#891](https://github.com/open-mmlab/mmpose/pull/891))

**Improvements**

- Add automatic benchmark regression tools ([#849](https://github.com/open-mmlab/mmpose/pull/849), [#880](https://github.com/open-mmlab/mmpose/pull/880), [#885](https://github.com/open-mmlab/mmpose/pull/885)) @liqikai9, @ly015

- Add copyright information and checking hook ([#872](https://github.com/open-mmlab/mmpose/pull/872))

- Add PR template ([#875](https://github.com/open-mmlab/mmpose/pull/875)) @ly015

- Add citation information ([#876](https://github.com/open-mmlab/mmpose/pull/876)) @ly015

- Add python3.9 in CI ([#877](https://github.com/open-mmlab/mmpose/pull/877), [#883](https://github.com/open-mmlab/mmpose/pull/883)) @ly015

- Improve the quality of the documents ([#845](https://github.com/open-mmlab/mmpose/pull/845), [#845](https://github.com/open-mmlab/mmpose/pull/845), [#848](https://github.com/open-mmlab/mmpose/pull/848), [#867](https://github.com/open-mmlab/mmpose/pull/867), [#870](https://github.com/open-mmlab/mmpose/pull/870), [#873](https://github.com/open-mmlab/mmpose/pull/873), [#896](https://github.com/open-mmlab/mmpose/pull/896)) @jin-s13, @ly015, @zhiqwang

## **v0.17.0 (06/08/2021)**

**Highlights**

1. Support ["Lite-HRNet: A Lightweight High-Resolution Network"](https://arxiv.org/abs/2104.06403) CVPR'2021 ([#733](https://github.com/open-mmlab/mmpose/pull/733),[#800](https://github.com/open-mmlab/mmpose/pull/800)) @jin-s13

2. Add 3d body mesh demo ([#771](https://github.com/open-mmlab/mmpose/pull/771)) @zengwang430521

3. Add Chinese documentation ([#787](https://github.com/open-mmlab/mmpose/pull/787), [#798](https://github.com/open-mmlab/mmpose/pull/798), [#799](https://github.com/open-mmlab/mmpose/pull/799), [#802](https://github.com/open-mmlab/mmpose/pull/802), [#804](https://github.com/open-mmlab/mmpose/pull/804), [#805](https://github.com/open-mmlab/mmpose/pull/805), [#815](https://github.com/open-mmlab/mmpose/pull/815), [#816](https://github.com/open-mmlab/mmpose/pull/816), [#817](https://github.com/open-mmlab/mmpose/pull/817), [#819](https://github.com/open-mmlab/mmpose/pull/819), [#839](https://github.com/open-mmlab/mmpose/pull/839)) @ly015, @luminxu, @jin-s13, @liqikai9, @zengwang430521

4. Add Colab Tutorial ([#834](https://github.com/open-mmlab/mmpose/pull/834)) @ly015

**New Features**

- Support ["Lite-HRNet: A Lightweight High-Resolution Network"](https://arxiv.org/abs/2104.06403) CVPR'2021 ([#733](https://github.com/open-mmlab/mmpose/pull/733),[#800](https://github.com/open-mmlab/mmpose/pull/800)) @jin-s13

- Add 3d body mesh demo ([#771](https://github.com/open-mmlab/mmpose/pull/771)) @zengwang430521

- Add Chinese documentation ([#787](https://github.com/open-mmlab/mmpose/pull/787), [#798](https://github.com/open-mmlab/mmpose/pull/798), [#799](https://github.com/open-mmlab/mmpose/pull/799), [#802](https://github.com/open-mmlab/mmpose/pull/802), [#804](https://github.com/open-mmlab/mmpose/pull/804), [#805](https://github.com/open-mmlab/mmpose/pull/805), [#815](https://github.com/open-mmlab/mmpose/pull/815), [#816](https://github.com/open-mmlab/mmpose/pull/816), [#817](https://github.com/open-mmlab/mmpose/pull/817), [#819](https://github.com/open-mmlab/mmpose/pull/819), [#839](https://github.com/open-mmlab/mmpose/pull/839)) @ly015, @luminxu, @jin-s13, @liqikai9, @zengwang430521

- Add Colab Tutorial ([#834](https://github.com/open-mmlab/mmpose/pull/834)) @ly015

- Support training for InterHand v1.0 dataset ([#761](https://github.com/open-mmlab/mmpose/pull/761)) @zengwang430521

**Bug Fixes**

- Fix mpii pckh@0.1 index ([#773](https://github.com/open-mmlab/mmpose/pull/773)) @jin-s13

- Fix multi-node distributed test ([#818](https://github.com/open-mmlab/mmpose/pull/818)) @ly015

- Fix docstring and init_weights error of ShuffleNetV1 ([#814](https://github.com/open-mmlab/mmpose/pull/814)) @Junjun2016

- Fix imshow_bbox error when input bboxes is empty ([#796](https://github.com/open-mmlab/mmpose/pull/796)) @ly015

- Fix model zoo doc generation ([#778](https://github.com/open-mmlab/mmpose/pull/778)) @ly015

- Fix typo ([#767](https://github.com/open-mmlab/mmpose/pull/767)), ([#780](https://github.com/open-mmlab/mmpose/pull/780), [#782](https://github.com/open-mmlab/mmpose/pull/782)) @ly015, @jin-s13

**Breaking Changes**

- Use MMCV EvalHook ([#686](https://github.com/open-mmlab/mmpose/pull/686)) @ly015

**Improvements**

- Add pytest.ini and fix docstring ([#812](https://github.com/open-mmlab/mmpose/pull/812)) @jin-s13

- Update MSELoss ([#829](https://github.com/open-mmlab/mmpose/pull/829)) @Ezra-Yu

- Move process_mmdet_results into inference.py ([#831](https://github.com/open-mmlab/mmpose/pull/831)) @ly015

- Update resource limit ([#783](https://github.com/open-mmlab/mmpose/pull/783)) @jin-s13

- Use COCO 2D pose model in 3D demo examples ([#785](https://github.com/open-mmlab/mmpose/pull/785)) @ly015

- Change model zoo titles in the doc from center-aligned to left-aligned ([#792](https://github.com/open-mmlab/mmpose/pull/792), [#797](https://github.com/open-mmlab/mmpose/pull/797)) @ly015

- Support MIM ([#706](https://github.com/open-mmlab/mmpose/pull/706), [#794](https://github.com/open-mmlab/mmpose/pull/794)) @ly015

- Update out-of-date configs ([#827](https://github.com/open-mmlab/mmpose/pull/827)) @jin-s13

- Remove opencv-python-headless dependency by albumentations ([#833](https://github.com/open-mmlab/mmpose/pull/833)) @ly015

- Update QQ QR code in README_CN.md ([#832](https://github.com/open-mmlab/mmpose/pull/832)) @ly015

## **v0.16.0 (02/07/2021)**

**Highlights**

1. Support ["ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"](https://arxiv.org/abs/2105.10154) CVPR'2021 ([#742](https://github.com/open-mmlab/mmpose/pull/742),[#755](https://github.com/open-mmlab/mmpose/pull/755)).

2. Support MPI-INF-3DHP dataset ([#683](https://github.com/open-mmlab/mmpose/pull/683),[#746](https://github.com/open-mmlab/mmpose/pull/746),[#751](https://github.com/open-mmlab/mmpose/pull/751)).

3. Add webcam demo tool ([#729](https://github.com/open-mmlab/mmpose/pull/729))

4. Add 3d body and hand pose estimation demo ([#704](https://github.com/open-mmlab/mmpose/pull/704), [#727](https://github.com/open-mmlab/mmpose/pull/727)).

**New Features**

- Support ["ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"](https://arxiv.org/abs/2105.10154) CVPR'2021 ([#742](https://github.com/open-mmlab/mmpose/pull/742),[#755](https://github.com/open-mmlab/mmpose/pull/755))

- Support MPI-INF-3DHP dataset ([#683](https://github.com/open-mmlab/mmpose/pull/683),[#746](https://github.com/open-mmlab/mmpose/pull/746),[#751](https://github.com/open-mmlab/mmpose/pull/751))

- Support Webcam demo ([#729](https://github.com/open-mmlab/mmpose/pull/729))

- Support Interhand 3d demo ([#704](https://github.com/open-mmlab/mmpose/pull/704))

- Support 3d pose video demo ([#727](https://github.com/open-mmlab/mmpose/pull/727))

- Support H36m dataset for 2d pose estimation ([#709](https://github.com/open-mmlab/mmpose/pull/709), [#735](https://github.com/open-mmlab/mmpose/pull/735))

- Add scripts to generate mim metafile ([#749](https://github.com/open-mmlab/mmpose/pull/749))

**Bug Fixes**

- Fix typos ([#692](https://github.com/open-mmlab/mmpose/pull/692),[#696](https://github.com/open-mmlab/mmpose/pull/696),[#697](https://github.com/open-mmlab/mmpose/pull/697),[#698](https://github.com/open-mmlab/mmpose/pull/698),[#712](https://github.com/open-mmlab/mmpose/pull/712),[#718](https://github.com/open-mmlab/mmpose/pull/718),[#728](https://github.com/open-mmlab/mmpose/pull/728))

- Change model download links from `http` to `https` ([#716](https://github.com/open-mmlab/mmpose/pull/716))

**Breaking Changes**

- Switch to MMCV MODEL_REGISTRY ([#669](https://github.com/open-mmlab/mmpose/pull/669))

**Improvements**

- Refactor MeshMixDataset ([#752](https://github.com/open-mmlab/mmpose/pull/752))

- Rename 'GaussianHeatMap' to 'GaussianHeatmap' ([#745](https://github.com/open-mmlab/mmpose/pull/745))

- Update out-of-date configs ([#734](https://github.com/open-mmlab/mmpose/pull/734))

- Improve compatibility for breaking changes ([#731](https://github.com/open-mmlab/mmpose/pull/731))

- Enable to control radius and thickness in visualization ([#722](https://github.com/open-mmlab/mmpose/pull/722))

- Add regex dependency ([#720](https://github.com/open-mmlab/mmpose/pull/720))

## **v0.15.0 (02/06/2021)**

**Highlights**

1. Support 3d video pose estimation (VideoPose3D).

2. Support 3d hand pose estimation (InterNet).

3. Improve presentation of modelzoo.

**New Features**

- Support "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image" (ECCV‘20) ([#624](https://github.com/open-mmlab/mmpose/pull/624))

- Support "3D human pose estimation in video with temporal convolutions and semi-supervised training" (CVPR'19)  ([#602](https://github.com/open-mmlab/mmpose/pull/602), [#681](https://github.com/open-mmlab/mmpose/pull/681))

- Support 3d pose estimation demo ([#653](https://github.com/open-mmlab/mmpose/pull/653), [#670](https://github.com/open-mmlab/mmpose/pull/670))

- Support bottom-up whole-body pose estimation ([#689](https://github.com/open-mmlab/mmpose/pull/689))

- Support mmcli ([#634](https://github.com/open-mmlab/mmpose/pull/634))

**Bug Fixes**

- Fix opencv compatibility ([#635](https://github.com/open-mmlab/mmpose/pull/635))

- Fix demo with UDP ([#637](https://github.com/open-mmlab/mmpose/pull/637))

- Fix bottom-up model onnx conversion ([#680](https://github.com/open-mmlab/mmpose/pull/680))

- Fix `GPU_IDS` in distributed training ([#668](https://github.com/open-mmlab/mmpose/pull/668))

- Fix MANIFEST.in ([#641](https://github.com/open-mmlab/mmpose/pull/641), [#657](https://github.com/open-mmlab/mmpose/pull/657))

- Fix docs ([#643](https://github.com/open-mmlab/mmpose/pull/643),[#684](https://github.com/open-mmlab/mmpose/pull/684),[#688](https://github.com/open-mmlab/mmpose/pull/688),[#690](https://github.com/open-mmlab/mmpose/pull/690),[#692](https://github.com/open-mmlab/mmpose/pull/692))

**Breaking Changes**

- Reorganize configs by tasks, algorithms, datasets, and techniques ([#647](https://github.com/open-mmlab/mmpose/pull/647))

- Rename heads and detectors ([#667](https://github.com/open-mmlab/mmpose/pull/667))

**Improvements**

- Add `radius` and `thickness` parameters in visualization ([#638](https://github.com/open-mmlab/mmpose/pull/638))

- Add `trans_prob` parameter in `TopDownRandomTranslation` ([#650](https://github.com/open-mmlab/mmpose/pull/650))

- Switch to `MMCV MODEL_REGISTRY` ([#669](https://github.com/open-mmlab/mmpose/pull/669))

- Update dependencies ([#674](https://github.com/open-mmlab/mmpose/pull/674), [#676](https://github.com/open-mmlab/mmpose/pull/676))

## **v0.14.0 (06/05/2021)**

**Highlights**

1. Support animal pose estimation with 7 popular datasets.

2. Support "A simple yet effective baseline for 3d human pose estimation" (ICCV'17).

**New Features**

- Support "A simple yet effective baseline for 3d human pose estimation" (ICCV'17)  ([#554](https://github.com/open-mmlab/mmpose/pull/554),[#558](https://github.com/open-mmlab/mmpose/pull/558),[#566](https://github.com/open-mmlab/mmpose/pull/566),[#570](https://github.com/open-mmlab/mmpose/pull/570),[#589](https://github.com/open-mmlab/mmpose/pull/589))

- Support animal pose estimation ([#559](https://github.com/open-mmlab/mmpose/pull/559),[#561](https://github.com/open-mmlab/mmpose/pull/561),[#563](https://github.com/open-mmlab/mmpose/pull/563),[#571](https://github.com/open-mmlab/mmpose/pull/571),[#603](https://github.com/open-mmlab/mmpose/pull/603),[#605](https://github.com/open-mmlab/mmpose/pull/605))

- Support Horse-10 dataset ([#561](https://github.com/open-mmlab/mmpose/pull/561)), MacaquePose dataset ([#561](https://github.com/open-mmlab/mmpose/pull/561)), Vinegar Fly dataset ([#561](https://github.com/open-mmlab/mmpose/pull/561)), Desert Locust dataset ([#561](https://github.com/open-mmlab/mmpose/pull/561)), Grevy's Zebra dataset ([#561](https://github.com/open-mmlab/mmpose/pull/561)), ATRW dataset ([#571](https://github.com/open-mmlab/mmpose/pull/571)), and Animal-Pose dataset ([#603](https://github.com/open-mmlab/mmpose/pull/603))

- Support bottom-up pose tracking demo ([#574](https://github.com/open-mmlab/mmpose/pull/574))

- Support FP16 training ([#584](https://github.com/open-mmlab/mmpose/pull/584),[#616](https://github.com/open-mmlab/mmpose/pull/616),[#626](https://github.com/open-mmlab/mmpose/pull/626))

- Support NMS for bottom-up ([#609](https://github.com/open-mmlab/mmpose/pull/609))

**Bug Fixes**

- Fix bugs in the top-down demo, when there are no people in the images ([#569](https://github.com/open-mmlab/mmpose/pull/569)).

- Fix the links in the doc ([#612](https://github.com/open-mmlab/mmpose/pull/612))

**Improvements**

- Speed up top-down inference ([#560](https://github.com/open-mmlab/mmpose/pull/560))

- Update github CI ([#562](https://github.com/open-mmlab/mmpose/pull/562), [#564](https://github.com/open-mmlab/mmpose/pull/564))

- Update Readme ([#578](https://github.com/open-mmlab/mmpose/pull/578),[#579](https://github.com/open-mmlab/mmpose/pull/579),[#580](https://github.com/open-mmlab/mmpose/pull/580),[#592](https://github.com/open-mmlab/mmpose/pull/592),[#599](https://github.com/open-mmlab/mmpose/pull/599),[#600](https://github.com/open-mmlab/mmpose/pull/600),[#607](https://github.com/open-mmlab/mmpose/pull/607))

- Update Faq ([#587](https://github.com/open-mmlab/mmpose/pull/587), [#610](https://github.com/open-mmlab/mmpose/pull/610))

## **v0.13.0 (31/03/2021)**

**Highlights**

1. Support Wingloss.

2. Support RHD hand dataset.

**New Features**

- Support Wingloss ([#482](https://github.com/open-mmlab/mmpose/pull/482))

- Support RHD hand dataset ([#523](https://github.com/open-mmlab/mmpose/pull/523), [#551](https://github.com/open-mmlab/mmpose/pull/551))

- Support Human3.6m dataset for 3d keypoint detection ([#518](https://github.com/open-mmlab/mmpose/pull/518), [#527](https://github.com/open-mmlab/mmpose/pull/527))

- Support TCN model for 3d keypoint detection ([#521](https://github.com/open-mmlab/mmpose/pull/521), [#522](https://github.com/open-mmlab/mmpose/pull/522))

- Support Interhand3D model for 3d hand detection ([#536](https://github.com/open-mmlab/mmpose/pull/536))

- Support Multi-task detector ([#480](https://github.com/open-mmlab/mmpose/pull/480))

**Bug Fixes**

- Fix PCKh@0.1 calculation ([#516](https://github.com/open-mmlab/mmpose/pull/516))

- Fix unittest ([#529](https://github.com/open-mmlab/mmpose/pull/529))

- Fix circular importing ([#542](https://github.com/open-mmlab/mmpose/pull/542))

- Fix bugs in bottom-up keypoint score ([#548](https://github.com/open-mmlab/mmpose/pull/548))

**Improvements**

- Update config & checkpoints ([#525](https://github.com/open-mmlab/mmpose/pull/525), [#546](https://github.com/open-mmlab/mmpose/pull/546))

- Fix typos ([#514](https://github.com/open-mmlab/mmpose/pull/514), [#519](https://github.com/open-mmlab/mmpose/pull/519), [#532](https://github.com/open-mmlab/mmpose/pull/532), [#537](https://github.com/open-mmlab/mmpose/pull/537), )

- Speed up post processing ([#535](https://github.com/open-mmlab/mmpose/pull/535))

- Update mmcv version dependency ([#544](https://github.com/open-mmlab/mmpose/pull/544))

## **v0.12.0 (28/02/2021)**

**Highlights**

1. Support DeepPose algorithm.

**New Features**

- Support DeepPose algorithm ([#446](https://github.com/open-mmlab/mmpose/pull/446), [#461](https://github.com/open-mmlab/mmpose/pull/461))

- Support interhand3d dataset ([#468](https://github.com/open-mmlab/mmpose/pull/468))

- Support Albumentation pipeline ([#469](https://github.com/open-mmlab/mmpose/pull/469))

- Support PhotometricDistortion pipeline ([#485](https://github.com/open-mmlab/mmpose/pull/485))

- Set seed option for training ([#493](https://github.com/open-mmlab/mmpose/pull/493))

- Add demos for face keypoint detection ([#502](https://github.com/open-mmlab/mmpose/pull/502))

**Bug Fixes**

- Change channel order according to configs ([#504](https://github.com/open-mmlab/mmpose/pull/504))

- Fix `num_factors` in UDP encoding ([#495](https://github.com/open-mmlab/mmpose/pull/495))

- Fix configs ([#456](https://github.com/open-mmlab/mmpose/pull/456))

**Breaking Changes**

- Refactor configs for wholebody pose estimation ([#487](https://github.com/open-mmlab/mmpose/pull/487), [#491](https://github.com/open-mmlab/mmpose/pull/491))

- Rename `decode` function for heads ([#481](https://github.com/open-mmlab/mmpose/pull/481))

**Improvements**

- Update config & checkpoints ([#453](https://github.com/open-mmlab/mmpose/pull/453),[#484](https://github.com/open-mmlab/mmpose/pull/484),[#487](https://github.com/open-mmlab/mmpose/pull/487))

- Add README in Chinese ([#462](https://github.com/open-mmlab/mmpose/pull/462))

- Add tutorials about configs  ([#465](https://github.com/open-mmlab/mmpose/pull/465))

- Add demo videos for various tasks ([#499](https://github.com/open-mmlab/mmpose/pull/499), [#503](https://github.com/open-mmlab/mmpose/pull/503))

- Update docs about MMPose installation ([#467](https://github.com/open-mmlab/mmpose/pull/467), [#505](https://github.com/open-mmlab/mmpose/pull/505))

- Rename `stat.py` to `stats.py` ([#483](https://github.com/open-mmlab/mmpose/pull/483))

- Fix typos ([#463](https://github.com/open-mmlab/mmpose/pull/463), [#464](https://github.com/open-mmlab/mmpose/pull/464), [#477](https://github.com/open-mmlab/mmpose/pull/477), [#481](https://github.com/open-mmlab/mmpose/pull/481))

- latex to bibtex ([#471](https://github.com/open-mmlab/mmpose/pull/471))

- Update FAQ ([#466](https://github.com/open-mmlab/mmpose/pull/466))

## **v0.11.0 (31/01/2021)**

**Highlights**

1. Support fashion landmark detection.

2. Support face keypoint detection.

3. Support pose tracking with MMTracking.

**New Features**

- Support fashion landmark detection (DeepFashion) ([#413](https://github.com/open-mmlab/mmpose/pull/413))

- Support face keypoint detection (300W, AFLW, COFW, WFLW) ([#367](https://github.com/open-mmlab/mmpose/pull/367))

- Support pose tracking demo with MMTracking ([#427](https://github.com/open-mmlab/mmpose/pull/427))

- Support face demo ([#443](https://github.com/open-mmlab/mmpose/pull/443))

- Support AIC dataset for bottom-up methods ([#438](https://github.com/open-mmlab/mmpose/pull/438), [#449](https://github.com/open-mmlab/mmpose/pull/449))

**Bug Fixes**

- Fix multi-batch training ([#434](https://github.com/open-mmlab/mmpose/pull/434))

- Fix sigmas in AIC dataset ([#441](https://github.com/open-mmlab/mmpose/pull/441))

- Fix config file ([#420](https://github.com/open-mmlab/mmpose/pull/420))

**Breaking Changes**

- Refactor Heads ([#382](https://github.com/open-mmlab/mmpose/pull/382))

**Improvements**

- Update readme ([#409](https://github.com/open-mmlab/mmpose/pull/409), [#412](https://github.com/open-mmlab/mmpose/pull/412), [#415](https://github.com/open-mmlab/mmpose/pull/415), [#416](https://github.com/open-mmlab/mmpose/pull/416), [#419](https://github.com/open-mmlab/mmpose/pull/419), [#421](https://github.com/open-mmlab/mmpose/pull/421), [#422](https://github.com/open-mmlab/mmpose/pull/422), [#424](https://github.com/open-mmlab/mmpose/pull/424), [#425](https://github.com/open-mmlab/mmpose/pull/425), [#435](https://github.com/open-mmlab/mmpose/pull/435), [#436](https://github.com/open-mmlab/mmpose/pull/436), [#437](https://github.com/open-mmlab/mmpose/pull/437), [#444](https://github.com/open-mmlab/mmpose/pull/444), [#445](https://github.com/open-mmlab/mmpose/pull/445))

- Add GAP (global average pooling) neck ([#414](https://github.com/open-mmlab/mmpose/pull/414))

- Speed up ([#411](https://github.com/open-mmlab/mmpose/pull/411), [#423](https://github.com/open-mmlab/mmpose/pull/423))

- Support COCO test-dev test ([#433](https://github.com/open-mmlab/mmpose/pull/433))

## **v0.10.0 (31/12/2020)**

**Highlights**

1. Support more human pose estimation methods.

   1. [UDP](https://arxiv.org/abs/1911.07524)

2. Support pose tracking.

3. Support multi-batch inference.

4. Add some useful tools, including `analyze_logs`, `get_flops`, `print_config`.

5. Support more backbone networks.

   1. [ResNest](https://arxiv.org/pdf/2004.08955.pdf)
   2. [VGG](https://arxiv.org/abs/1409.1556)

**New Features**

- Support UDP ([#353](https://github.com/open-mmlab/mmpose/pull/353), [#371](https://github.com/open-mmlab/mmpose/pull/371), [#402](https://github.com/open-mmlab/mmpose/pull/402))

- Support multi-batch inference ([#390](https://github.com/open-mmlab/mmpose/pull/390))

- Support MHP dataset ([#386](https://github.com/open-mmlab/mmpose/pull/386))

- Support pose tracking demo ([#380](https://github.com/open-mmlab/mmpose/pull/380))

- Support mpii-trb demo ([#372](https://github.com/open-mmlab/mmpose/pull/372))

- Support mobilenet for hand pose estimation ([#377](https://github.com/open-mmlab/mmpose/pull/377))

- Support ResNest backbone ([#370](https://github.com/open-mmlab/mmpose/pull/370))

- Support VGG backbone ([#370](https://github.com/open-mmlab/mmpose/pull/370))

- Add some useful tools, including `analyze_logs`, `get_flops`, `print_config` ([#324](https://github.com/open-mmlab/mmpose/pull/324))

**Bug Fixes**

- Fix bugs in pck evaluation ([#328](https://github.com/open-mmlab/mmpose/pull/328))

- Fix model download links in README ([#396](https://github.com/open-mmlab/mmpose/pull/396), [#397](https://github.com/open-mmlab/mmpose/pull/397))

- Fix CrowdPose annotations and update benchmarks ([#384](https://github.com/open-mmlab/mmpose/pull/384))

- Fix modelzoo stat ([#354](https://github.com/open-mmlab/mmpose/pull/354), [#360](https://github.com/open-mmlab/mmpose/pull/360), [#362](https://github.com/open-mmlab/mmpose/pull/362))

- Fix config files for aic datasets ([#340](https://github.com/open-mmlab/mmpose/pull/340))

**Breaking Changes**

- Rename `image_thr` to `det_bbox_thr` for top-down methods.

**Improvements**

- Organize the readme files ([#398](https://github.com/open-mmlab/mmpose/pull/398), [#399](https://github.com/open-mmlab/mmpose/pull/399), [#400](https://github.com/open-mmlab/mmpose/pull/400))

- Check linting for markdown ([#379](https://github.com/open-mmlab/mmpose/pull/379))

- Add faq.md ([#350](https://github.com/open-mmlab/mmpose/pull/350))

- Remove PyTorch 1.4 in CI ([#338](https://github.com/open-mmlab/mmpose/pull/338))

- Add pypi badge in readme ([#329](https://github.com/open-mmlab/mmpose/pull/329))

## **v0.9.0 (30/11/2020)**

**Highlights**

1. Support more human pose estimation methods.

   1. [MSPN](https://arxiv.org/abs/1901.00148)
   2. [RSN](https://arxiv.org/abs/2003.04030)

2. Support video pose estimation datasets.

   1. [sub-JHMDB](http://jhmdb.is.tue.mpg.de/dataset)

3. Support Onnx model conversion.

**New Features**

- Support MSPN ([#278](https://github.com/open-mmlab/mmpose/pull/278))

- Support RSN ([#221](https://github.com/open-mmlab/mmpose/pull/221), [#318](https://github.com/open-mmlab/mmpose/pull/318))

- Support new post-processing method for MSPN & RSN ([#288](https://github.com/open-mmlab/mmpose/pull/288))

- Support sub-JHMDB dataset ([#292](https://github.com/open-mmlab/mmpose/pull/292))

- Support urls for pre-trained models in config files ([#232](https://github.com/open-mmlab/mmpose/pull/232))

- Support Onnx ([#305](https://github.com/open-mmlab/mmpose/pull/305))

**Bug Fixes**

- Fix model download links in README ([#255](https://github.com/open-mmlab/mmpose/pull/255), [#315](https://github.com/open-mmlab/mmpose/pull/315))

**Breaking Changes**

- `post_process=True|False` and `unbiased_decoding=True|False` are deprecated, use `post_process=None|default|unbiased` etc. instead ([#288](https://github.com/open-mmlab/mmpose/pull/288))

**Improvements**

- Enrich the model zoo ([#256](https://github.com/open-mmlab/mmpose/pull/256), [#320](https://github.com/open-mmlab/mmpose/pull/320))

- Set the default map_location as 'cpu' to reduce gpu memory cost ([#227](https://github.com/open-mmlab/mmpose/pull/227))

- Support return heatmaps and backbone features for bottom-up models ([#229](https://github.com/open-mmlab/mmpose/pull/229))

- Upgrade mmcv maximum & minimum version ([#269](https://github.com/open-mmlab/mmpose/pull/269), [#313](https://github.com/open-mmlab/mmpose/pull/313))

- Automatically add modelzoo statistics to readthedocs ([#252](https://github.com/open-mmlab/mmpose/pull/252))

- Fix Pylint issues ([#258](https://github.com/open-mmlab/mmpose/pull/258), [#259](https://github.com/open-mmlab/mmpose/pull/259), [#260](https://github.com/open-mmlab/mmpose/pull/260), [#262](https://github.com/open-mmlab/mmpose/pull/262), [#265](https://github.com/open-mmlab/mmpose/pull/265), [#267](https://github.com/open-mmlab/mmpose/pull/267), [#268](https://github.com/open-mmlab/mmpose/pull/268), [#270](https://github.com/open-mmlab/mmpose/pull/270), [#271](https://github.com/open-mmlab/mmpose/pull/271), [#272](https://github.com/open-mmlab/mmpose/pull/272), [#273](https://github.com/open-mmlab/mmpose/pull/273), [#275](https://github.com/open-mmlab/mmpose/pull/275), [#276](https://github.com/open-mmlab/mmpose/pull/276), [#283](https://github.com/open-mmlab/mmpose/pull/283), [#285](https://github.com/open-mmlab/mmpose/pull/285), [#293](https://github.com/open-mmlab/mmpose/pull/293), [#294](https://github.com/open-mmlab/mmpose/pull/294), [#295](https://github.com/open-mmlab/mmpose/pull/295))

- Improve README ([#226](https://github.com/open-mmlab/mmpose/pull/226), [#257](https://github.com/open-mmlab/mmpose/pull/257), [#264](https://github.com/open-mmlab/mmpose/pull/264), [#280](https://github.com/open-mmlab/mmpose/pull/280), [#296](https://github.com/open-mmlab/mmpose/pull/296))

- Support PyTorch 1.7 in CI ([#274](https://github.com/open-mmlab/mmpose/pull/274))

- Add docs/tutorials for running demos ([#263](https://github.com/open-mmlab/mmpose/pull/263))

## **v0.8.0 (31/10/2020)**

**Highlights**

1. Support more human pose estimation datasets.

   1. [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)
   2. [PoseTrack18](https://posetrack.net/)

2. Support more 2D hand keypoint estimation datasets.

   1. [InterHand2.6](https://github.com/facebookresearch/InterHand2.6M)

3. Support adversarial training for 3D human shape recovery.

4. Support multi-stage losses.

5. Support mpii demo.

**New Features**

- Support [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) dataset ([#195](https://github.com/open-mmlab/mmpose/pull/195))

- Support [PoseTrack18](https://posetrack.net/) dataset ([#220](https://github.com/open-mmlab/mmpose/pull/220))

- Support [InterHand2.6](https://github.com/facebookresearch/InterHand2.6M) dataset ([#202](https://github.com/open-mmlab/mmpose/pull/202))

- Support adversarial training for 3D human shape recovery ([#192](https://github.com/open-mmlab/mmpose/pull/192))

- Support multi-stage losses ([#204](https://github.com/open-mmlab/mmpose/pull/204))

**Bug Fixes**

- Fix config files ([#190](https://github.com/open-mmlab/mmpose/pull/190))

**Improvements**

- Add mpii demo ([#216](https://github.com/open-mmlab/mmpose/pull/216))

- Improve README ([#181](https://github.com/open-mmlab/mmpose/pull/181), [#183](https://github.com/open-mmlab/mmpose/pull/183), [#208](https://github.com/open-mmlab/mmpose/pull/208))

- Support return heatmaps and backbone features ([#196](https://github.com/open-mmlab/mmpose/pull/196), [#212](https://github.com/open-mmlab/mmpose/pull/212))

- Support different return formats of mmdetection models ([#217](https://github.com/open-mmlab/mmpose/pull/217))

## **v0.7.0 (30/9/2020)**

**Highlights**

1. Support HMR for 3D human shape recovery.

2. Support WholeBody human pose estimation.

   1. [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody)

3. Support more 2D hand keypoint estimation datasets.

   1. [Frei-hand](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
   2. [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html)

4. Add more popular backbones & enrich the [modelzoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html)

   1. ShuffleNetv2

5. Support hand demo and whole-body demo.

**New Features**

- Support HMR for 3D human shape recovery ([#157](https://github.com/open-mmlab/mmpose/pull/157), [#160](https://github.com/open-mmlab/mmpose/pull/160), [#161](https://github.com/open-mmlab/mmpose/pull/161), [#162](https://github.com/open-mmlab/mmpose/pull/162))

- Support [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) dataset ([#133](https://github.com/open-mmlab/mmpose/pull/133))

- Support [Frei-hand](https://lmb.informatik.uni-freiburg.de/projects/freihand/) dataset ([#125](https://github.com/open-mmlab/mmpose/pull/125))

- Support [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html) dataset ([#144](https://github.com/open-mmlab/mmpose/pull/144))

- Support H36M dataset ([#159](https://github.com/open-mmlab/mmpose/pull/159))

- Support ShuffleNetv2 ([#139](https://github.com/open-mmlab/mmpose/pull/139))

- Support saving best models based on key indicator ([#127](https://github.com/open-mmlab/mmpose/pull/127))

**Bug Fixes**

- Fix typos in docs ([#121](https://github.com/open-mmlab/mmpose/pull/121))

- Fix assertion ([#142](https://github.com/open-mmlab/mmpose/pull/142))

**Improvements**

- Add tools to transform .mat format to .json format ([#126](https://github.com/open-mmlab/mmpose/pull/126))

- Add hand demo ([#115](https://github.com/open-mmlab/mmpose/pull/115))

- Add whole-body demo ([#163](https://github.com/open-mmlab/mmpose/pull/163))

- Reuse mmcv utility function and update version files ([#135](https://github.com/open-mmlab/mmpose/pull/135), [#137](https://github.com/open-mmlab/mmpose/pull/137))

- Enrich the modelzoo ([#147](https://github.com/open-mmlab/mmpose/pull/147), [#169](https://github.com/open-mmlab/mmpose/pull/169))

- Improve docs ([#174](https://github.com/open-mmlab/mmpose/pull/174), [#175](https://github.com/open-mmlab/mmpose/pull/175), [#178](https://github.com/open-mmlab/mmpose/pull/178))

- Improve README ([#176](https://github.com/open-mmlab/mmpose/pull/176))

- Improve version.py ([#173](https://github.com/open-mmlab/mmpose/pull/173))

## **v0.6.0 (31/8/2020)**

**Highlights**

1. Add more popular backbones & enrich the [modelzoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html)

   1. ResNext
   2. SEResNet
   3. ResNetV1D
   4. MobileNetv2
   5. ShuffleNetv1
   6. CPM (Convolutional Pose Machine)

2. Add more popular datasets:

   1. [AIChallenger](https://arxiv.org/abs/1711.06475?context=cs.CV)
   2. [MPII](http://human-pose.mpi-inf.mpg.de/)
   3. [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)
   4. [OCHuman](http://www.liruilong.cn/projects/pose2seg/index.html)

3. Support 2d hand keypoint estimation.

   1. [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)

4. Support bottom-up inference.

**New Features**

- Support [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html) dataset ([#52](https://github.com/open-mmlab/mmpose/pull/52))

- Support [MPII](http://human-pose.mpi-inf.mpg.de/) dataset ([#55](https://github.com/open-mmlab/mmpose/pull/55))

- Support [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) dataset ([#19](https://github.com/open-mmlab/mmpose/pull/19), [#47](https://github.com/open-mmlab/mmpose/pull/47), [#48](https://github.com/open-mmlab/mmpose/pull/48))

- Support [OCHuman](http://www.liruilong.cn/projects/pose2seg/index.html) dataset ([#70](https://github.com/open-mmlab/mmpose/pull/70))

- Support [AIChallenger](https://arxiv.org/abs/1711.06475?context=cs.CV) dataset ([#87](https://github.com/open-mmlab/mmpose/pull/87))

- Support multiple backbones ([#26](https://github.com/open-mmlab/mmpose/pull/26))

- Support CPM model ([#56](https://github.com/open-mmlab/mmpose/pull/56))

**Bug Fixes**

- Fix configs for MPII & MPII-TRB datasets ([#93](https://github.com/open-mmlab/mmpose/pull/93))

- Fix the bug of missing `test_pipeline` in configs ([#14](https://github.com/open-mmlab/mmpose/pull/14))

- Fix typos ([#27](https://github.com/open-mmlab/mmpose/pull/27), [#28](https://github.com/open-mmlab/mmpose/pull/28), [#50](https://github.com/open-mmlab/mmpose/pull/50), [#53](https://github.com/open-mmlab/mmpose/pull/53), [#63](https://github.com/open-mmlab/mmpose/pull/63))

**Improvements**

- Update benchmark ([#93](https://github.com/open-mmlab/mmpose/pull/93))

- Add Dockerfile ([#44](https://github.com/open-mmlab/mmpose/pull/44))

- Improve unittest coverage and minor fix ([#18](https://github.com/open-mmlab/mmpose/pull/18))

- Support CPUs for train/val/demo ([#34](https://github.com/open-mmlab/mmpose/pull/34))

- Support bottom-up demo ([#69](https://github.com/open-mmlab/mmpose/pull/69))

- Add tools to publish model ([#62](https://github.com/open-mmlab/mmpose/pull/62))

- Enrich the modelzoo ([#64](https://github.com/open-mmlab/mmpose/pull/64), [#68](https://github.com/open-mmlab/mmpose/pull/68), [#82](https://github.com/open-mmlab/mmpose/pull/82))

## **v0.5.0 (21/7/2020)**

**Highlights**

- MMPose is released.

**Main Features**

- Support both top-down and bottom-up pose estimation approaches.

- Achieve higher training efficiency and higher accuracy than other popular codebases (e.g. AlphaPose, HRNet)

- Support various backbone models: ResNet, HRNet, SCNet, Houglass and HigherHRNet.
