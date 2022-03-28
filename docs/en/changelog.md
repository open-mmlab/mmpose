# Changelog

## v0.24.0 (07/03/2022)

**Highlights**

- Support HRFormer ["HRFormer: High-Resolution Vision Transformer for Dense Predict"](https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html), NeurIPS'2021 ([\#1203](https://github.com/open-mmlab/mmpose/pull/1203)) @zengwang430521
- Support Windows installation with pip ([\#1213](https://github.com/open-mmlab/mmpose/pull/1213)) @jin-s13, @ly015
- Add WebcamAPI documents ([\#1187](https://github.com/open-mmlab/mmpose/pull/1187)) @ly015

**New Features**

- Support HRFormer ["HRFormer: High-Resolution Vision Transformer for Dense Predict"](https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html), NeurIPS'2021 ([\#1203](https://github.com/open-mmlab/mmpose/pull/1203)) @zengwang430521
- Support Windows installation with pip ([\#1213](https://github.com/open-mmlab/mmpose/pull/1213)) @jin-s13, @ly015
- Support CPU training with mmcv < v1.4.4 ([\#1161](https://github.com/open-mmlab/mmpose/pull/1161)) @EasonQYS, @ly015
- Add "Valentine Magic" demo with WebcamAPI ([\#1189](https://github.com/open-mmlab/mmpose/pull/1189), [\#1191](https://github.com/open-mmlab/mmpose/pull/1191)) @liqikai9

**Improvements**

- Refactor multi-view 3D pose estimation framework towards better modularization and expansibility ([\#1196](https://github.com/open-mmlab/mmpose/pull/1196)) @wusize
- Add WebcamAPI documents and tutorials ([\#1187](https://github.com/open-mmlab/mmpose/pull/1187)) @ly015
- Refactor dataset evaluation interface to align with other OpenMMLab codebases ([\#1209](https://github.com/open-mmlab/mmpose/pull/1209)) @ly015
- Add deprecation message for deploy tools since [MMDeploy](https://github.com/open-mmlab/mmdeploy) has supported MMPose ([\#1207](https://github.com/open-mmlab/mmpose/pull/1207)) @QwQ2000
- Improve documentation quality ([\#1206](https://github.com/open-mmlab/mmpose/pull/1206), [\#1161](https://github.com/open-mmlab/mmpose/pull/1161)) @ly015
- Switch to OpenMMLab official pre-commit-hook for copyright check ([\#1214](https://github.com/open-mmlab/mmpose/pull/1214)) @ly015

**Bug Fixes**

- Fix hard-coded data collating and scattering in inference ([\#1175](https://github.com/open-mmlab/mmpose/pull/1175)) @ly015
- Fix model configs on JHMDB dataset ([\#1188](https://github.com/open-mmlab/mmpose/pull/1188)) @jin-s13
- Fix area calculation in pose tracking inference ([\#1197](https://github.com/open-mmlab/mmpose/pull/1197)) @pallgeuer
- Fix registry scope conflict of module wrapper ([\#1204](https://github.com/open-mmlab/mmpose/pull/1204)) @ly015
- Update MMCV installation in CI and documents ([\#1205](https://github.com/open-mmlab/mmpose/pull/1205))
- Fix incorrect color channel order in visualization functions ([\#1212](https://github.com/open-mmlab/mmpose/pull/1212)) @ly015

## v0.23.0 (11/02/2022)

**Highlights**

- Add [MMPose Webcam API](https://github.com/open-mmlab/mmpose/tree/master/tools/webcam): A simple yet powerful tools to develop interactive webcam applications with MMPose functions. ([\#1178](https://github.com/open-mmlab/mmpose/pull/1178), [\#1173](https://github.com/open-mmlab/mmpose/pull/1173), [\#1173](https://github.com/open-mmlab/mmpose/pull/1173), [\#1143](https://github.com/open-mmlab/mmpose/pull/1143), [\#1094](https://github.com/open-mmlab/mmpose/pull/1094), [\#1133](https://github.com/open-mmlab/mmpose/pull/1133), [\#1098](https://github.com/open-mmlab/mmpose/pull/1098), [\#1160](https://github.com/open-mmlab/mmpose/pull/1160)) @ly015, @jin-s13, @liqikai9, @wusize, @luminxu, @zengwang430521 @mzr1996

**New Features**

- Add [MMPose Webcam API](https://github.com/open-mmlab/mmpose/tree/master/tools/webcam): A simple yet powerful tools to develop interactive webcam applications with MMPose functions. ([\#1178](https://github.com/open-mmlab/mmpose/pull/1178), [\#1173](https://github.com/open-mmlab/mmpose/pull/1173), [\#1173](https://github.com/open-mmlab/mmpose/pull/1173), [\#1143](https://github.com/open-mmlab/mmpose/pull/1143), [\#1094](https://github.com/open-mmlab/mmpose/pull/1094), [\#1133](https://github.com/open-mmlab/mmpose/pull/1133), [\#1098](https://github.com/open-mmlab/mmpose/pull/1098), [\#1160](https://github.com/open-mmlab/mmpose/pull/1160)) @ly015, @jin-s13, @liqikai9, @wusize, @luminxu, @zengwang430521 @mzr1996
- Support ConcatDataset ([\#1139](https://github.com/open-mmlab/mmpose/pull/1139)) @Canwang-sjtu
- Support CPU training and testing ([\#1157](https://github.com/open-mmlab/mmpose/pull/1157)) @ly015

**Improvements**

- Add multi-processing configurations to speed up distributed training and testing ([\#1146](https://github.com/open-mmlab/mmpose/pull/1146)) @ly015
- Add default runtime config ([\#1145](https://github.com/open-mmlab/mmpose/pull/1145))

- Upgrade isort in pre-commit hook ([\#1179](https://github.com/open-mmlab/mmpose/pull/1179)) @liqikai9
- Update README and documents ([\#1171](https://github.com/open-mmlab/mmpose/pull/1171), [\#1167](https://github.com/open-mmlab/mmpose/pull/1167), [\#1153](https://github.com/open-mmlab/mmpose/pull/1153), [\#1149](https://github.com/open-mmlab/mmpose/pull/1149), [\#1148](https://github.com/open-mmlab/mmpose/pull/1148), [\#1147](https://github.com/open-mmlab/mmpose/pull/1147), [\#1140](https://github.com/open-mmlab/mmpose/pull/1140)) @jin-s13, @wusize, @TommyZihao, @ly015

**Bug Fixes**

- Fix undeterministic behavior in pre-commit hooks ([\#1136](https://github.com/open-mmlab/mmpose/pull/1136)) @jin-s13
- Deprecate the support for "python setup.py test" ([\#1179](https://github.com/open-mmlab/mmpose/pull/1179)) @ly015
- Fix incompatible settings with MMCV on HSigmoid default parameters ([\#1132](https://github.com/open-mmlab/mmpose/pull/1132)) @ly015
- Fix albumentation installation ([\#1184](https://github.com/open-mmlab/mmpose/pull/1184)) @BIGWangYuDong

## v0.22.0 (04/01/2022)

**Highlights**

- Support VoxelPose ["VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment"](https://arxiv.org/abs/2004.06239), ECCV'2020 ([\#1050](https://github.com/open-mmlab/mmpose/pull/1050)) @wusize
- Support Soft Wing loss ["Structure-Coherent Deep Feature Learning for Robust Face Alignment"](https://linchunze.github.io/papers/TIP21_Structure_coherent_FA.pdf), TIP'2021 ([\#1077](https://github.com/open-mmlab/mmpose/pull/1077)) @jin-s13
- Support Adaptive Wing loss ["Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"](https://arxiv.org/abs/1904.07399), ICCV'2019 ([\#1072](https://github.com/open-mmlab/mmpose/pull/1072)) @jin-s13

**New Features**

- Support VoxelPose ["VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment"](https://arxiv.org/abs/2004.06239), ECCV'2020 ([\#1050](https://github.com/open-mmlab/mmpose/pull/1050)) @wusize
- Support Soft Wing loss ["Structure-Coherent Deep Feature Learning for Robust Face Alignment"](https://linchunze.github.io/papers/TIP21_Structure_coherent_FA.pdf), TIP'2021 ([\#1077](https://github.com/open-mmlab/mmpose/pull/1077)) @jin-s13
- Support Adaptive Wing loss ["Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"](https://arxiv.org/abs/1904.07399), ICCV'2019 ([\#1072](https://github.com/open-mmlab/mmpose/pull/1072)) @jin-s13
- Add LiteHRNet-18 Checkpoints trained on COCO. ([\#1120](https://github.com/open-mmlab/mmpose/pull/1120)) @jin-s13

**Improvements**

- Improve documentation quality ([\#1115](https://github.com/open-mmlab/mmpose/pull/1115), [\#1111](https://github.com/open-mmlab/mmpose/pull/1111), [\#1105](https://github.com/open-mmlab/mmpose/pull/1105), [\#1087](https://github.com/open-mmlab/mmpose/pull/1087), [\#1086](https://github.com/open-mmlab/mmpose/pull/1086), [\#1085](https://github.com/open-mmlab/mmpose/pull/1085), [\#1084](https://github.com/open-mmlab/mmpose/pull/1084), [\#1083](https://github.com/open-mmlab/mmpose/pull/1083), [\#1124](https://github.com/open-mmlab/mmpose/pull/1124), [\#1070](https://github.com/open-mmlab/mmpose/pull/1070), [\#1068](https://github.com/open-mmlab/mmpose/pull/1068)) @jin-s13, @liqikai9, @ly015
- Support CircleCI ([\#1074](https://github.com/open-mmlab/mmpose/pull/1074)) @ly015
- Skip unit tests in CI when only document files were changed ([\#1074](https://github.com/open-mmlab/mmpose/pull/1074), [\#1041](https://github.com/open-mmlab/mmpose/pull/1041)) @QwQ2000, @ly015
- Support file_client_args in LoadImageFromFile ([\#1076](https://github.com/open-mmlab/mmpose/pull/1076)) @jin-s13

**Bug Fixes**

- Fix a bug in Dark UDP postprocessing that causes error when the channel number is large. ([\#1079](https://github.com/open-mmlab/mmpose/pull/1079), [\#1116](https://github.com/open-mmlab/mmpose/pull/1116)) @X00123, @jin-s13
- Fix hard-coded `sigmas` in bottom-up image demo ([\#1107](https://github.com/open-mmlab/mmpose/pull/1107), [\#1101](https://github.com/open-mmlab/mmpose/pull/1101)) @chenxinfeng4, @liqikai9
- Fix unstable checks in unit tests ([\#1112](https://github.com/open-mmlab/mmpose/pull/1112)) @ly015
- Do not destroy NULL windows if `args.show==False` in demo scripts ([\#1104](https://github.com/open-mmlab/mmpose/pull/1104)) @bladrome

## v0.21.0 (06/12/2021)

**Highlights**

- Support ["Learning Temporal Pose Estimation from Sparsely-Labeled Videos"](https://arxiv.org/abs/1906.04016), NeurIPS'2019 ([\#932](https://github.com/open-mmlab/mmpose/pull/932), [\#1006](https://github.com/open-mmlab/mmpose/pull/1006), [\#1036](https://github.com/open-mmlab/mmpose/pull/1036), [\#1060](https://github.com/open-mmlab/mmpose/pull/1060)) @liqikai9
- Add ViPNAS-MobileNetV3 models ([\#1025](https://github.com/open-mmlab/mmpose/pull/1025)) @luminxu, @jin-s13
- Add [inference speed benchmark](/docs/en/inference_speed_summary.md) ([\#1028](https://github.com/open-mmlab/mmpose/pull/1028), [\#1034](https://github.com/open-mmlab/mmpose/pull/1034), [\#1044](https://github.com/open-mmlab/mmpose/pull/1044)) @liqikai9

**New Features**

- Support ["Learning Temporal Pose Estimation from Sparsely-Labeled Videos"](https://arxiv.org/abs/1906.04016), NeurIPS'2019 ([\#932](https://github.com/open-mmlab/mmpose/pull/932), [\#1006](https://github.com/open-mmlab/mmpose/pull/1006), [\#1036](https://github.com/open-mmlab/mmpose/pull/1036)) @liqikai9
- Add ViPNAS-MobileNetV3 models ([\#1025](https://github.com/open-mmlab/mmpose/pull/1025)) @luminxu, @jin-s13
- Add light-weight top-down models for whole-body keypoint detection ([\#1009](https://github.com/open-mmlab/mmpose/pull/1009), [\#1020](https://github.com/open-mmlab/mmpose/pull/1020), [\#1055](https://github.com/open-mmlab/mmpose/pull/1055)) @luminxu, @ly015
- Add HRNet checkpoints with various settings on PoseTrack18 ([\#1035](https://github.com/open-mmlab/mmpose/pull/1035)) @liqikai9

**Improvements**

- Add [inference speed benchmark](/docs/en/inference_speed_summary.md) ([\#1028](https://github.com/open-mmlab/mmpose/pull/1028), [\#1034](https://github.com/open-mmlab/mmpose/pull/1034), [\#1044](https://github.com/open-mmlab/mmpose/pull/1044)) @liqikai9
- Update model metafile format ([\#1001](https://github.com/open-mmlab/mmpose/pull/1001)) @ly015
- Support minus output feature index in mobilenet_v3 ([\#1005](https://github.com/open-mmlab/mmpose/pull/1005)) @luminxu
- Improve documentation quality ([\#1018](https://github.com/open-mmlab/mmpose/pull/1018), [\#1026](https://github.com/open-mmlab/mmpose/pull/1026), [\#1027](https://github.com/open-mmlab/mmpose/pull/1027), [\#1031](https://github.com/open-mmlab/mmpose/pull/1031), [\#1038](https://github.com/open-mmlab/mmpose/pull/1038), [\#1046](https://github.com/open-mmlab/mmpose/pull/1046), [\#1056](https://github.com/open-mmlab/mmpose/pull/1056), [\#1057](https://github.com/open-mmlab/mmpose/pull/1057)) @edybk, @luminxu, @ly015, @jin-s13
- Set default random seed in training initialization ([\#1030](https://github.com/open-mmlab/mmpose/pull/1030)) @ly015
- Skip CI when only specific files changed ([\#1041](https://github.com/open-mmlab/mmpose/pull/1041), [\#1059](https://github.com/open-mmlab/mmpose/pull/1059)) @QwQ2000, @ly015
- Automatically cancel uncompleted action runs when new commit arrives ([\#1053](https://github.com/open-mmlab/mmpose/pull/1053)) @ly015

**Bug Fixes**

- Update pose tracking demo to be compatible with latest mmtracking ([\#1014](https://github.com/open-mmlab/mmpose/pull/1014)) @jin-s13
- Fix symlink creation failure when installed in Windows environments ([\#1039](https://github.com/open-mmlab/mmpose/pull/1039)) @QwQ2000
- Fix AP-10K dataset sigmas ([\#1040](https://github.com/open-mmlab/mmpose/pull/1040)) @jin-s13

## v0.20.0 (01/11/2021)

**Highlights**

- Add AP-10K dataset for animal pose estimation ([\#987](https://github.com/open-mmlab/mmpose/pull/987)) @Annbless, @AlexTheBad, @jin-s13, @ly015
- Support TorchServe ([\#979](https://github.com/open-mmlab/mmpose/pull/979)) @ly015

**New Features**

- Add AP-10K dataset for animal pose estimation ([\#987](https://github.com/open-mmlab/mmpose/pull/987)) @Annbless, @AlexTheBad, @jin-s13, @ly015
- Add HRNetv2 checkpoints on 300W and COFW datasets ([\#980](https://github.com/open-mmlab/mmpose/pull/980)) @jin-s13
- Support TorchServe ([\#979](https://github.com/open-mmlab/mmpose/pull/979)) @ly015

**Bug Fixes**

- Fix some deprecated or risky settings in configs ([\#963](https://github.com/open-mmlab/mmpose/pull/963), [\#976](https://github.com/open-mmlab/mmpose/pull/976), [\#992](https://github.com/open-mmlab/mmpose/pull/992)) @jin-s13, @wusize
- Fix issues of default arguments of training and testing scripts ([\#970](https://github.com/open-mmlab/mmpose/pull/970), [\#985](https://github.com/open-mmlab/mmpose/pull/985)) @liqikai9, @wusize
- Fix heatmap and tag size mismatch in bottom-up with UDP ([\#994](https://github.com/open-mmlab/mmpose/pull/994)) @wusize
- Fix python3.9 installation in CI ([\#983](https://github.com/open-mmlab/mmpose/pull/983)) @ly015
- Fix model zoo document integrity issue ([\#990](https://github.com/open-mmlab/mmpose/pull/990)) @jin-s13

**Improvements**

- Support non-square input shape for bottom-up ([\#991](https://github.com/open-mmlab/mmpose/pull/991)) @wusize
- Add image and video resources for demo ([\#971](https://github.com/open-mmlab/mmpose/pull/971)) @liqikai9
- Use CUDA docker images to accelerate CI ([\#973](https://github.com/open-mmlab/mmpose/pull/973)) @ly015
- Add codespell hook and fix detected typos ([\#977](https://github.com/open-mmlab/mmpose/pull/977)) @ly015

## v0.19.0 (08/10/2021)

**Highlights**

- Add models for Associative Embedding with Hourglass network backbone ([\#906](https://github.com/open-mmlab/mmpose/pull/906), [\#955](https://github.com/open-mmlab/mmpose/pull/955)) @jin-s13, @luminxu
- Support COCO-Wholebody-Face and COCO-Wholebody-Hand datasets ([\#813](https://github.com/open-mmlab/mmpose/pull/813)) @jin-s13, @innerlee, @luminxu
- Upgrade dataset interface ([\#901](https://github.com/open-mmlab/mmpose/pull/901), [\#924](https://github.com/open-mmlab/mmpose/pull/924)) @jin-s13, @innerlee, @ly015, @liqikai9
- New style of documentation ([\#945](https://github.com/open-mmlab/mmpose/pull/945)) @ly015

**New Features**

- Add models for Associative Embedding with Hourglass network backbone ([\#906](https://github.com/open-mmlab/mmpose/pull/906), [\#955](https://github.com/open-mmlab/mmpose/pull/955)) @jin-s13, @luminxu
- Support COCO-Wholebody-Face and COCO-Wholebody-Hand datasets ([\#813](https://github.com/open-mmlab/mmpose/pull/813)) @jin-s13, @innerlee, @luminxu
- Add pseudo-labeling tool to generate COCO style keypoint annotations with given bounding boxes ([\#928](https://github.com/open-mmlab/mmpose/pull/928)) @soltkreig
- New style of documentation ([\#945](https://github.com/open-mmlab/mmpose/pull/945)) @ly015

**Bug Fixes**

- Fix segmentation parsing in Macaque dataset preprocessing ([\#948](https://github.com/open-mmlab/mmpose/pull/948)) @jin-s13
- Fix dependencies that may lead to CI failure in downstream projects ([\#936](https://github.com/open-mmlab/mmpose/pull/936), [\#953](https://github.com/open-mmlab/mmpose/pull/953)) @RangiLyu, @ly015
- Fix keypoint order in Human3.6M dataset ([\#940](https://github.com/open-mmlab/mmpose/pull/940)) @ttxskk
- Fix unstable image loading for Interhand2.6M ([\#913](https://github.com/open-mmlab/mmpose/pull/913)) @zengwang430521

**Improvements**

- Upgrade dataset interface ([\#901](https://github.com/open-mmlab/mmpose/pull/901), [\#924](https://github.com/open-mmlab/mmpose/pull/924)) @jin-s13, @innerlee, @ly015, @liqikai9
- Improve demo usability and stability ([\#908](https://github.com/open-mmlab/mmpose/pull/908), [\#934](https://github.com/open-mmlab/mmpose/pull/934)) @ly015
- Standardize model metafile format ([\#941](https://github.com/open-mmlab/mmpose/pull/941)) @ly015
- Support `persistent_worker` and several other arguments in configs ([\#946](https://github.com/open-mmlab/mmpose/pull/946)) @jin-s13
- Use MMCV root model registry to enable cross-project module building ([\#935](https://github.com/open-mmlab/mmpose/pull/935)) @RangiLyu
- Improve the document quality ([\#916](https://github.com/open-mmlab/mmpose/pull/916), [\#909](https://github.com/open-mmlab/mmpose/pull/909), [\#942](https://github.com/open-mmlab/mmpose/pull/942), [\#913](https://github.com/open-mmlab/mmpose/pull/913), [\#956](https://github.com/open-mmlab/mmpose/pull/956)) @jin-s13, @ly015, @bit-scientist, @zengwang430521
- Improve pull request template ([\#952](https://github.com/open-mmlab/mmpose/pull/952), [\#954](https://github.com/open-mmlab/mmpose/pull/954)) @ly015

**Breaking Changes**

- Upgrade dataset interface ([\#901](https://github.com/open-mmlab/mmpose/pull/901)) @jin-s13, @innerlee, @ly015

## v0.18.0 (01/09/2021)

**Bug Fixes**

- Fix redundant model weight loading in pytorch-to-onnx conversion ([\#850](https://github.com/open-mmlab/mmpose/pull/850)) @ly015
- Fix a bug in update_model_index.py that may cause pre-commit hook failure([\#866](https://github.com/open-mmlab/mmpose/pull/866)) @ly015
- Fix a bug in interhand_3d_head ([\#890](https://github.com/open-mmlab/mmpose/pull/890)) @zengwang430521
- Fix pose tracking demo failure caused by out-of-date configs ([\#891](https://github.com/open-mmlab/mmpose/pull/891))

**Improvements**

- Add automatic benchmark regression tools ([\#849](https://github.com/open-mmlab/mmpose/pull/849), [\#880](https://github.com/open-mmlab/mmpose/pull/880), [\#885](https://github.com/open-mmlab/mmpose/pull/885)) @liqikai9, @ly015
- Add copyright information and checking hook ([\#872](https://github.com/open-mmlab/mmpose/pull/872))
- Add PR template ([\#875](https://github.com/open-mmlab/mmpose/pull/875)) @ly015
- Add citation information ([\#876](https://github.com/open-mmlab/mmpose/pull/876)) @ly015
- Add python3.9 in CI ([\#877](https://github.com/open-mmlab/mmpose/pull/877), [\#883](https://github.com/open-mmlab/mmpose/pull/883)) @ly015
- Improve the quality of the documents ([\#845](https://github.com/open-mmlab/mmpose/pull/845), [\#845](https://github.com/open-mmlab/mmpose/pull/845), [\#848](https://github.com/open-mmlab/mmpose/pull/848), [\#867](https://github.com/open-mmlab/mmpose/pull/867), [\#870](https://github.com/open-mmlab/mmpose/pull/870), [\#873](https://github.com/open-mmlab/mmpose/pull/873), [\#896](https://github.com/open-mmlab/mmpose/pull/896)) @jin-s13, @ly015, @zhiqwang

## v0.17.0 (06/08/2021)

**Highlights**

1. Support ["Lite-HRNet: A Lightweight High-Resolution Network"](https://arxiv.org/abs/2104.06403) CVPR'2021 ([\#733](https://github.com/open-mmlab/mmpose/pull/733),[\#800](https://github.com/open-mmlab/mmpose/pull/800)) @jin-s13
2. Add 3d body mesh demo ([\#771](https://github.com/open-mmlab/mmpose/pull/771)) @zengwang430521
3. Add Chinese documentation ([\#787](https://github.com/open-mmlab/mmpose/pull/787), [\#798](https://github.com/open-mmlab/mmpose/pull/798), [\#799](https://github.com/open-mmlab/mmpose/pull/799), [\#802](https://github.com/open-mmlab/mmpose/pull/802), [\#804](https://github.com/open-mmlab/mmpose/pull/804), [\#805](https://github.com/open-mmlab/mmpose/pull/805), [\#815](https://github.com/open-mmlab/mmpose/pull/815), [\#816](https://github.com/open-mmlab/mmpose/pull/816), [\#817](https://github.com/open-mmlab/mmpose/pull/817), [\#819](https://github.com/open-mmlab/mmpose/pull/819), [\#839](https://github.com/open-mmlab/mmpose/pull/839)) @ly015, @luminxu, @jin-s13, @liqikai9, @zengwang430521
4. Add Colab Tutorial ([\#834](https://github.com/open-mmlab/mmpose/pull/834)) @ly015

**New Features**

- Support ["Lite-HRNet: A Lightweight High-Resolution Network"](https://arxiv.org/abs/2104.06403) CVPR'2021 ([\#733](https://github.com/open-mmlab/mmpose/pull/733),[\#800](https://github.com/open-mmlab/mmpose/pull/800)) @jin-s13
- Add 3d body mesh demo ([\#771](https://github.com/open-mmlab/mmpose/pull/771)) @zengwang430521
- Add Chinese documentation ([\#787](https://github.com/open-mmlab/mmpose/pull/787), [\#798](https://github.com/open-mmlab/mmpose/pull/798), [\#799](https://github.com/open-mmlab/mmpose/pull/799), [\#802](https://github.com/open-mmlab/mmpose/pull/802), [\#804](https://github.com/open-mmlab/mmpose/pull/804), [\#805](https://github.com/open-mmlab/mmpose/pull/805), [\#815](https://github.com/open-mmlab/mmpose/pull/815), [\#816](https://github.com/open-mmlab/mmpose/pull/816), [\#817](https://github.com/open-mmlab/mmpose/pull/817), [\#819](https://github.com/open-mmlab/mmpose/pull/819), [\#839](https://github.com/open-mmlab/mmpose/pull/839)) @ly015, @luminxu, @jin-s13, @liqikai9, @zengwang430521
- Add Colab Tutorial ([\#834](https://github.com/open-mmlab/mmpose/pull/834)) @ly015
- Support training for InterHand v1.0 dataset ([\#761](https://github.com/open-mmlab/mmpose/pull/761)) @zengwang430521

**Bug Fixes**

- Fix mpii pckh@0.1 index ([\#773](https://github.com/open-mmlab/mmpose/pull/773)) @jin-s13
- Fix multi-node distributed test ([\#818](https://github.com/open-mmlab/mmpose/pull/818)) @ly015
- Fix docstring and init_weights error of ShuffleNetV1 ([\#814](https://github.com/open-mmlab/mmpose/pull/814)) @Junjun2016
- Fix imshow_bbox error when input bboxes is empty ([\#796](https://github.com/open-mmlab/mmpose/pull/796)) @ly015
- Fix model zoo doc generation ([\#778](https://github.com/open-mmlab/mmpose/pull/778)) @ly015
- Fix typo ([\#767](https://github.com/open-mmlab/mmpose/pull/767)), ([\#780](https://github.com/open-mmlab/mmpose/pull/780), [\#782](https://github.com/open-mmlab/mmpose/pull/782)) @ly015, @jin-s13

**Breaking Changes**

- Use MMCV EvalHook ([\#686](https://github.com/open-mmlab/mmpose/pull/686)) @ly015

**Improvements**

- Add pytest.ini and fix docstring ([\#812](https://github.com/open-mmlab/mmpose/pull/812)) @jin-s13
- Update MSELoss ([\#829](https://github.com/open-mmlab/mmpose/pull/829)) @Ezra-Yu
- Move process_mmdet_results into inference.py ([\#831](https://github.com/open-mmlab/mmpose/pull/831)) @ly015
- Update resource limit ([\#783](https://github.com/open-mmlab/mmpose/pull/783)) @jin-s13
- Use COCO 2D pose model in 3D demo examples ([\#785](https://github.com/open-mmlab/mmpose/pull/785)) @ly015
- Change model zoo titles in the doc from center-aligned to left-aligned ([\#792](https://github.com/open-mmlab/mmpose/pull/792), [\#797](https://github.com/open-mmlab/mmpose/pull/797)) @ly015
- Support MIM ([\#706](https://github.com/open-mmlab/mmpose/pull/706), [\#794](https://github.com/open-mmlab/mmpose/pull/794)) @ly015
- Update out-of-date configs ([\#827](https://github.com/open-mmlab/mmpose/pull/827)) @jin-s13
- Remove opencv-python-headless dependency by albumentations ([\#833](https://github.com/open-mmlab/mmpose/pull/833)) @ly015
- Update QQ QR code in README_CN.md ([\#832](https://github.com/open-mmlab/mmpose/pull/832)) @ly015

## v0.16.0 (02/07/2021)

**Highlights**

1. Support ["ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"](https://arxiv.org/abs/2105.10154) CVPR'2021 ([\#742](https://github.com/open-mmlab/mmpose/pull/742),[\#755](https://github.com/open-mmlab/mmpose/pull/755)).
1. Support MPI-INF-3DHP dataset ([\#683](https://github.com/open-mmlab/mmpose/pull/683),[\#746](https://github.com/open-mmlab/mmpose/pull/746),[\#751](https://github.com/open-mmlab/mmpose/pull/751)).
1. Add webcam demo tool ([\#729](https://github.com/open-mmlab/mmpose/pull/729))
1. Add 3d body and hand pose estimation demo ([\#704](https://github.com/open-mmlab/mmpose/pull/704), [\#727](https://github.com/open-mmlab/mmpose/pull/727)).

**New Features**

- Support ["ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search"](https://arxiv.org/abs/2105.10154) CVPR'2021 ([\#742](https://github.com/open-mmlab/mmpose/pull/742),[\#755](https://github.com/open-mmlab/mmpose/pull/755))
- Support MPI-INF-3DHP dataset ([\#683](https://github.com/open-mmlab/mmpose/pull/683),[\#746](https://github.com/open-mmlab/mmpose/pull/746),[\#751](https://github.com/open-mmlab/mmpose/pull/751))
- Support Webcam demo ([\#729](https://github.com/open-mmlab/mmpose/pull/729))
- Support Interhand 3d demo ([\#704](https://github.com/open-mmlab/mmpose/pull/704))
- Support 3d pose video demo ([\#727](https://github.com/open-mmlab/mmpose/pull/727))
- Support H36m dataset for 2d pose estimation ([\#709](https://github.com/open-mmlab/mmpose/pull/709), [\#735](https://github.com/open-mmlab/mmpose/pull/735))
- Add scripts to generate mim metafile ([\#749](https://github.com/open-mmlab/mmpose/pull/749))

**Bug Fixes**

- Fix typos ([\#692](https://github.com/open-mmlab/mmpose/pull/692),[\#696](https://github.com/open-mmlab/mmpose/pull/696),[\#697](https://github.com/open-mmlab/mmpose/pull/697),[\#698](https://github.com/open-mmlab/mmpose/pull/698),[\#712](https://github.com/open-mmlab/mmpose/pull/712),[\#718](https://github.com/open-mmlab/mmpose/pull/718),[\#728](https://github.com/open-mmlab/mmpose/pull/728))
- Change model download links from `http` to `https` ([\#716](https://github.com/open-mmlab/mmpose/pull/716))

**Breaking Changes**

- Switch to MMCV MODEL_REGISTRY ([\#669](https://github.com/open-mmlab/mmpose/pull/669))

**Improvements**

- Refactor MeshMixDataset ([\#752](https://github.com/open-mmlab/mmpose/pull/752))
- Rename 'GaussianHeatMap' to 'GaussianHeatmap' ([\#745](https://github.com/open-mmlab/mmpose/pull/745))
- Update out-of-date configs ([\#734](https://github.com/open-mmlab/mmpose/pull/734))
- Improve compatibility for breaking changes ([\#731](https://github.com/open-mmlab/mmpose/pull/731))
- Enable to control radius and thickness in visualization ([\#722](https://github.com/open-mmlab/mmpose/pull/722))
- Add regex dependency ([\#720](https://github.com/open-mmlab/mmpose/pull/720))

## v0.15.0 (02/06/2021)

**Highlights**

1. Support 3d video pose estimation (VideoPose3D).
1. Support 3d hand pose estimation (InterNet).
1. Improve presentation of modelzoo.

**New Features**

- Support "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image" (ECCV‘20) ([\#624](https://github.com/open-mmlab/mmpose/pull/624))
- Support "3D human pose estimation in video with temporal convolutions and semi-supervised training" (CVPR'19)  ([\#602](https://github.com/open-mmlab/mmpose/pull/602), [\#681](https://github.com/open-mmlab/mmpose/pull/681))
- Support 3d pose estimation demo ([\#653](https://github.com/open-mmlab/mmpose/pull/653), [\#670](https://github.com/open-mmlab/mmpose/pull/670))
- Support bottom-up whole-body pose estimation ([\#689](https://github.com/open-mmlab/mmpose/pull/689))
- Support mmcli ([\#634](https://github.com/open-mmlab/mmpose/pull/634))

**Bug Fixes**

- Fix opencv compatibility ([\#635](https://github.com/open-mmlab/mmpose/pull/635))
- Fix demo with UDP ([\#637](https://github.com/open-mmlab/mmpose/pull/637))
- Fix bottom-up model onnx conversion ([\#680](https://github.com/open-mmlab/mmpose/pull/680))
- Fix `GPU_IDS` in distributed training ([\#668](https://github.com/open-mmlab/mmpose/pull/668))
- Fix MANIFEST.in ([\#641](https://github.com/open-mmlab/mmpose/pull/641), [\#657](https://github.com/open-mmlab/mmpose/pull/657))
- Fix docs ([\#643](https://github.com/open-mmlab/mmpose/pull/643),[\#684](https://github.com/open-mmlab/mmpose/pull/684),[\#688](https://github.com/open-mmlab/mmpose/pull/688),[\#690](https://github.com/open-mmlab/mmpose/pull/690),[\#692](https://github.com/open-mmlab/mmpose/pull/692))

**Breaking Changes**

- Reorganize configs by tasks, algorithms, datasets, and techniques ([\#647](https://github.com/open-mmlab/mmpose/pull/647))
- Rename heads and detectors ([\#667](https://github.com/open-mmlab/mmpose/pull/667))

**Improvements**

- Add `radius` and `thickness` parameters in visualization ([\#638](https://github.com/open-mmlab/mmpose/pull/638))
- Add `trans_prob` parameter in `TopDownRandomTranslation` ([\#650](https://github.com/open-mmlab/mmpose/pull/650))
- Switch to `MMCV MODEL_REGISTRY` ([\#669](https://github.com/open-mmlab/mmpose/pull/669))
- Update dependencies ([\#674](https://github.com/open-mmlab/mmpose/pull/674), [\#676](https://github.com/open-mmlab/mmpose/pull/676))

## v0.14.0 (06/05/2021)

**Highlights**

1. Support animal pose estimation with 7 popular datasets.
1. Support "A simple yet effective baseline for 3d human pose estimation" (ICCV'17).

**New Features**

- Support "A simple yet effective baseline for 3d human pose estimation" (ICCV'17)  ([\#554](https://github.com/open-mmlab/mmpose/pull/554),[\#558](https://github.com/open-mmlab/mmpose/pull/558),[\#566](https://github.com/open-mmlab/mmpose/pull/566),[\#570](https://github.com/open-mmlab/mmpose/pull/570),[\#589](https://github.com/open-mmlab/mmpose/pull/589))
- Support animal pose estimation ([\#559](https://github.com/open-mmlab/mmpose/pull/559),[\#561](https://github.com/open-mmlab/mmpose/pull/561),[\#563](https://github.com/open-mmlab/mmpose/pull/563),[\#571](https://github.com/open-mmlab/mmpose/pull/571),[\#603](https://github.com/open-mmlab/mmpose/pull/603),[\#605](https://github.com/open-mmlab/mmpose/pull/605))
- Support Horse-10 dataset ([\#561](https://github.com/open-mmlab/mmpose/pull/561)), MacaquePose dataset ([\#561](https://github.com/open-mmlab/mmpose/pull/561)), Vinegar Fly dataset ([\#561](https://github.com/open-mmlab/mmpose/pull/561)), Desert Locust dataset ([\#561](https://github.com/open-mmlab/mmpose/pull/561)), Grevy's Zebra dataset ([\#561](https://github.com/open-mmlab/mmpose/pull/561)), ATRW dataset ([\#571](https://github.com/open-mmlab/mmpose/pull/571)), and Animal-Pose dataset ([\#603](https://github.com/open-mmlab/mmpose/pull/603))
- Support bottom-up pose tracking demo ([\#574](https://github.com/open-mmlab/mmpose/pull/574))
- Support FP16 training ([\#584](https://github.com/open-mmlab/mmpose/pull/584),[\#616](https://github.com/open-mmlab/mmpose/pull/616),[\#626](https://github.com/open-mmlab/mmpose/pull/626))
- Support NMS for bottom-up ([\#609](https://github.com/open-mmlab/mmpose/pull/609))

**Bug Fixes**

- Fix bugs in the top-down demo, when there are no people in the images ([\#569](https://github.com/open-mmlab/mmpose/pull/569)).
- Fix the links in the doc ([\#612](https://github.com/open-mmlab/mmpose/pull/612))

**Improvements**

- Speed up top-down inference ([\#560](https://github.com/open-mmlab/mmpose/pull/560))
- Update github CI ([\#562](https://github.com/open-mmlab/mmpose/pull/562), [\#564](https://github.com/open-mmlab/mmpose/pull/564))
- Update Readme ([\#578](https://github.com/open-mmlab/mmpose/pull/578),[\#579](https://github.com/open-mmlab/mmpose/pull/579),[\#580](https://github.com/open-mmlab/mmpose/pull/580),[\#592](https://github.com/open-mmlab/mmpose/pull/592),[\#599](https://github.com/open-mmlab/mmpose/pull/599),[\#600](https://github.com/open-mmlab/mmpose/pull/600),[\#607](https://github.com/open-mmlab/mmpose/pull/607))
- Update Faq ([\#587](https://github.com/open-mmlab/mmpose/pull/587), [\#610](https://github.com/open-mmlab/mmpose/pull/610))

## v0.13.0 (31/03/2021)

**Highlights**

1. Support Wingloss.
1. Support RHD hand dataset.

**New Features**

- Support Wingloss ([\#482](https://github.com/open-mmlab/mmpose/pull/482))
- Support RHD hand dataset ([\#523](https://github.com/open-mmlab/mmpose/pull/523), [\#551](https://github.com/open-mmlab/mmpose/pull/551))
- Support Human3.6m dataset for 3d keypoint detection ([\#518](https://github.com/open-mmlab/mmpose/pull/518), [\#527](https://github.com/open-mmlab/mmpose/pull/527))
- Support TCN model for 3d keypoint detection ([\#521](https://github.com/open-mmlab/mmpose/pull/521), [\#522](https://github.com/open-mmlab/mmpose/pull/522))
- Support Interhand3D model for 3d hand detection ([\#536](https://github.com/open-mmlab/mmpose/pull/536))
- Support Multi-task detector ([\#480](https://github.com/open-mmlab/mmpose/pull/480))

**Bug Fixes**

- Fix PCKh@0.1 calculation ([\#516](https://github.com/open-mmlab/mmpose/pull/516))
- Fix unittest ([\#529](https://github.com/open-mmlab/mmpose/pull/529))
- Fix circular importing ([\#542](https://github.com/open-mmlab/mmpose/pull/542))
- Fix bugs in bottom-up keypoint score ([\#548](https://github.com/open-mmlab/mmpose/pull/548))

**Improvements**

- Update config & checkpoints ([\#525](https://github.com/open-mmlab/mmpose/pull/525), [\#546](https://github.com/open-mmlab/mmpose/pull/546))
- Fix typos ([\#514](https://github.com/open-mmlab/mmpose/pull/514), [\#519](https://github.com/open-mmlab/mmpose/pull/519), [\#532](https://github.com/open-mmlab/mmpose/pull/532), [\#537](https://github.com/open-mmlab/mmpose/pull/537), )
- Speed up post processing ([\#535](https://github.com/open-mmlab/mmpose/pull/535))
- Update mmcv version dependency ([\#544](https://github.com/open-mmlab/mmpose/pull/544))

## v0.12.0 (28/02/2021)

**Highlights**

1. Support DeepPose algorithm.

**New Features**

- Support DeepPose algorithm ([\#446](https://github.com/open-mmlab/mmpose/pull/446), [\#461](https://github.com/open-mmlab/mmpose/pull/461))
- Support interhand3d dataset ([\#468](https://github.com/open-mmlab/mmpose/pull/468))
- Support Albumentation pipeline ([\#469](https://github.com/open-mmlab/mmpose/pull/469))
- Support PhotometricDistortion pipeline ([\#485](https://github.com/open-mmlab/mmpose/pull/485))
- Set seed option for training ([\#493](https://github.com/open-mmlab/mmpose/pull/493))
- Add demos for face keypoint detection ([\#502](https://github.com/open-mmlab/mmpose/pull/502))

**Bug Fixes**

- Change channel order according to configs ([\#504](https://github.com/open-mmlab/mmpose/pull/504))
- Fix `num_factors` in UDP encoding ([\#495](https://github.com/open-mmlab/mmpose/pull/495))
- Fix configs ([\#456](https://github.com/open-mmlab/mmpose/pull/456))

**Breaking Changes**

- Refactor configs for wholebody pose estimation ([\#487](https://github.com/open-mmlab/mmpose/pull/487), [\#491](https://github.com/open-mmlab/mmpose/pull/491))
- Rename `decode` function for heads ([\#481](https://github.com/open-mmlab/mmpose/pull/481))

**Improvements**

- Update config & checkpoints ([\#453](https://github.com/open-mmlab/mmpose/pull/453),[\#484](https://github.com/open-mmlab/mmpose/pull/484),[\#487](https://github.com/open-mmlab/mmpose/pull/487))
- Add README in Chinese ([\#462](https://github.com/open-mmlab/mmpose/pull/462))
- Add tutorials about configs  ([\#465](https://github.com/open-mmlab/mmpose/pull/465))
- Add demo videos for various tasks ([\#499](https://github.com/open-mmlab/mmpose/pull/499), [\#503](https://github.com/open-mmlab/mmpose/pull/503))
- Update docs about MMPose installation ([\#467](https://github.com/open-mmlab/mmpose/pull/467), [\#505](https://github.com/open-mmlab/mmpose/pull/505))
- Rename `stat.py` to `stats.py` ([\#483](https://github.com/open-mmlab/mmpose/pull/483))
- Fix typos ([\#463](https://github.com/open-mmlab/mmpose/pull/463), [\#464](https://github.com/open-mmlab/mmpose/pull/464), [\#477](https://github.com/open-mmlab/mmpose/pull/477), [\#481](https://github.com/open-mmlab/mmpose/pull/481))
- latex to bibtex ([\#471](https://github.com/open-mmlab/mmpose/pull/471))
- Update FAQ ([\#466](https://github.com/open-mmlab/mmpose/pull/466))

## v0.11.0 (31/01/2021)

**Highlights**

1. Support fashion landmark detection.
1. Support face keypoint detection.
1. Support pose tracking with MMTracking.

**New Features**

- Support fashion landmark detection (DeepFashion) ([\#413](https://github.com/open-mmlab/mmpose/pull/413))
- Support face keypoint detection (300W, AFLW, COFW, WFLW) ([\#367](https://github.com/open-mmlab/mmpose/pull/367))
- Support pose tracking demo with MMTracking ([\#427](https://github.com/open-mmlab/mmpose/pull/427))
- Support face demo ([\#443](https://github.com/open-mmlab/mmpose/pull/443))
- Support AIC dataset for bottom-up methods ([\#438](https://github.com/open-mmlab/mmpose/pull/438), [\#449](https://github.com/open-mmlab/mmpose/pull/449))

**Bug Fixes**

- Fix multi-batch training ([\#434](https://github.com/open-mmlab/mmpose/pull/434))
- Fix sigmas in AIC dataset ([\#441](https://github.com/open-mmlab/mmpose/pull/441))
- Fix config file ([\#420](https://github.com/open-mmlab/mmpose/pull/420))

**Breaking Changes**

- Refactor Heads ([\#382](https://github.com/open-mmlab/mmpose/pull/382))

**Improvements**

- Update readme ([\#409](https://github.com/open-mmlab/mmpose/pull/409), [\#412](https://github.com/open-mmlab/mmpose/pull/412), [\#415](https://github.com/open-mmlab/mmpose/pull/415), [\#416](https://github.com/open-mmlab/mmpose/pull/416), [\#419](https://github.com/open-mmlab/mmpose/pull/419), [\#421](https://github.com/open-mmlab/mmpose/pull/421), [\#422](https://github.com/open-mmlab/mmpose/pull/422), [\#424](https://github.com/open-mmlab/mmpose/pull/424), [\#425](https://github.com/open-mmlab/mmpose/pull/425), [\#435](https://github.com/open-mmlab/mmpose/pull/435), [\#436](https://github.com/open-mmlab/mmpose/pull/436), [\#437](https://github.com/open-mmlab/mmpose/pull/437), [\#444](https://github.com/open-mmlab/mmpose/pull/444), [\#445](https://github.com/open-mmlab/mmpose/pull/445))
- Add GAP (global average pooling) neck ([\#414](https://github.com/open-mmlab/mmpose/pull/414))
- Speed up ([\#411](https://github.com/open-mmlab/mmpose/pull/411), [\#423](https://github.com/open-mmlab/mmpose/pull/423))
- Support COCO test-dev test ([\#433](https://github.com/open-mmlab/mmpose/pull/433))

## v0.10.0 (31/12/2020)

**Highlights**

1. Support more human pose estimation methods.
   - [UDP](https://arxiv.org/abs/1911.07524)
1. Support pose tracking.
1. Support multi-batch inference.
1. Add some useful tools, including `analyze_logs`, `get_flops`, `print_config`.
1. Support more backbone networks.
   - [ResNest](https://arxiv.org/pdf/2004.08955.pdf)
   - [VGG](https://arxiv.org/abs/1409.1556)

**New Features**

- Support UDP ([\#353](https://github.com/open-mmlab/mmpose/pull/353), [\#371](https://github.com/open-mmlab/mmpose/pull/371), [\#402](https://github.com/open-mmlab/mmpose/pull/402))
- Support multi-batch inference ([\#390](https://github.com/open-mmlab/mmpose/pull/390))
- Support MHP dataset ([\#386](https://github.com/open-mmlab/mmpose/pull/386))
- Support pose tracking demo ([\#380](https://github.com/open-mmlab/mmpose/pull/380))
- Support mpii-trb demo ([\#372](https://github.com/open-mmlab/mmpose/pull/372))
- Support mobilenet for hand pose estimation ([\#377](https://github.com/open-mmlab/mmpose/pull/377))
- Support ResNest backbone ([\#370](https://github.com/open-mmlab/mmpose/pull/370))
- Support VGG backbone ([\#370](https://github.com/open-mmlab/mmpose/pull/370))
- Add some useful tools, including `analyze_logs`, `get_flops`, `print_config` ([\#324](https://github.com/open-mmlab/mmpose/pull/324))

**Bug Fixes**

- Fix bugs in pck evaluation ([\#328](https://github.com/open-mmlab/mmpose/pull/328))
- Fix model download links in README ([\#396](https://github.com/open-mmlab/mmpose/pull/396), [\#397](https://github.com/open-mmlab/mmpose/pull/397))
- Fix CrowdPose annotations and update benchmarks ([\#384](https://github.com/open-mmlab/mmpose/pull/384))
- Fix modelzoo stat ([\#354](https://github.com/open-mmlab/mmpose/pull/354), [\#360](https://github.com/open-mmlab/mmpose/pull/360), [\#362](https://github.com/open-mmlab/mmpose/pull/362))
- Fix config files for aic datasets ([\#340](https://github.com/open-mmlab/mmpose/pull/340))

**Breaking Changes**

- Rename `image_thr` to `det_bbox_thr` for top-down methods.

**Improvements**

- Organize the readme files ([\#398](https://github.com/open-mmlab/mmpose/pull/398), [\#399](https://github.com/open-mmlab/mmpose/pull/399), [\#400](https://github.com/open-mmlab/mmpose/pull/400))
- Check linting for markdown ([\#379](https://github.com/open-mmlab/mmpose/pull/379))
- Add faq.md ([\#350](https://github.com/open-mmlab/mmpose/pull/350))
- Remove PyTorch 1.4 in CI ([\#338](https://github.com/open-mmlab/mmpose/pull/338))
- Add pypi badge in readme ([\#329](https://github.com/open-mmlab/mmpose/pull/329))

## v0.9.0 (30/11/2020)

**Highlights**

1. Support more human pose estimation methods.
   - [MSPN](https://arxiv.org/abs/1901.00148)
   - [RSN](https://arxiv.org/abs/2003.04030)
1. Support video pose estimation datasets.
   - [sub-JHMDB](http://jhmdb.is.tue.mpg.de/dataset)
1. Support Onnx model conversion.

**New Features**

- Support MSPN ([\#278](https://github.com/open-mmlab/mmpose/pull/278))
- Support RSN ([\#221](https://github.com/open-mmlab/mmpose/pull/221), [\#318](https://github.com/open-mmlab/mmpose/pull/318))
- Support new post-processing method for MSPN & RSN ([\#288](https://github.com/open-mmlab/mmpose/pull/288))
- Support sub-JHMDB dataset ([\#292](https://github.com/open-mmlab/mmpose/pull/292))
- Support urls for pre-trained models in config files ([\#232](https://github.com/open-mmlab/mmpose/pull/232))
- Support Onnx ([\#305](https://github.com/open-mmlab/mmpose/pull/305))

**Bug Fixes**

- Fix model download links in README ([\#255](https://github.com/open-mmlab/mmpose/pull/255), [\#315](https://github.com/open-mmlab/mmpose/pull/315))

**Breaking Changes**

- `post_process=True|False` and `unbiased_decoding=True|False` are deprecated, use `post_process=None|default|unbiased` etc. instead ([\#288](https://github.com/open-mmlab/mmpose/pull/288))

**Improvements**

- Enrich the model zoo ([\#256](https://github.com/open-mmlab/mmpose/pull/256), [\#320](https://github.com/open-mmlab/mmpose/pull/320))
- Set the default map_location as 'cpu' to reduce gpu memory cost ([\#227](https://github.com/open-mmlab/mmpose/pull/227))
- Support return heatmaps and backbone features for bottom-up models ([\#229](https://github.com/open-mmlab/mmpose/pull/229))
- Upgrade mmcv maximum & minimum version ([\#269](https://github.com/open-mmlab/mmpose/pull/269), [\#313](https://github.com/open-mmlab/mmpose/pull/313))
- Automatically add modelzoo statistics to readthedocs ([\#252](https://github.com/open-mmlab/mmpose/pull/252))
- Fix Pylint issues ([\#258](https://github.com/open-mmlab/mmpose/pull/258), [\#259](https://github.com/open-mmlab/mmpose/pull/259), [\#260](https://github.com/open-mmlab/mmpose/pull/260), [\#262](https://github.com/open-mmlab/mmpose/pull/262), [\#265](https://github.com/open-mmlab/mmpose/pull/265), [\#267](https://github.com/open-mmlab/mmpose/pull/267), [\#268](https://github.com/open-mmlab/mmpose/pull/268), [\#270](https://github.com/open-mmlab/mmpose/pull/270), [\#271](https://github.com/open-mmlab/mmpose/pull/271), [\#272](https://github.com/open-mmlab/mmpose/pull/272), [\#273](https://github.com/open-mmlab/mmpose/pull/273), [\#275](https://github.com/open-mmlab/mmpose/pull/275), [\#276](https://github.com/open-mmlab/mmpose/pull/276), [\#283](https://github.com/open-mmlab/mmpose/pull/283), [\#285](https://github.com/open-mmlab/mmpose/pull/285), [\#293](https://github.com/open-mmlab/mmpose/pull/293), [\#294](https://github.com/open-mmlab/mmpose/pull/294), [\#295](https://github.com/open-mmlab/mmpose/pull/295))
- Improve README ([\#226](https://github.com/open-mmlab/mmpose/pull/226), [\#257](https://github.com/open-mmlab/mmpose/pull/257), [\#264](https://github.com/open-mmlab/mmpose/pull/264), [\#280](https://github.com/open-mmlab/mmpose/pull/280), [\#296](https://github.com/open-mmlab/mmpose/pull/296))
- Support PyTorch 1.7 in CI ([\#274](https://github.com/open-mmlab/mmpose/pull/274))
- Add docs/tutorials for running demos ([\#263](https://github.com/open-mmlab/mmpose/pull/263))

## v0.8.0 (31/10/2020)

**Highlights**

1. Support more human pose estimation datasets.
   - [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)
   - [PoseTrack18](https://posetrack.net/)
1. Support more 2D hand keypoint estimation datasets.
   - [InterHand2.6](https://github.com/facebookresearch/InterHand2.6M)
1. Support adversarial training for 3D human shape recovery.
1. Support multi-stage losses.
1. Support mpii demo.

**New Features**

- Support [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) dataset ([\#195](https://github.com/open-mmlab/mmpose/pull/195))
- Support [PoseTrack18](https://posetrack.net/) dataset ([\#220](https://github.com/open-mmlab/mmpose/pull/220))
- Support [InterHand2.6](https://github.com/facebookresearch/InterHand2.6M) dataset ([\#202](https://github.com/open-mmlab/mmpose/pull/202))
- Support adversarial training for 3D human shape recovery ([\#192](https://github.com/open-mmlab/mmpose/pull/192))
- Support multi-stage losses ([\#204](https://github.com/open-mmlab/mmpose/pull/204))

**Bug Fixes**

- Fix config files ([\#190](https://github.com/open-mmlab/mmpose/pull/190))

**Improvements**

- Add mpii demo ([\#216](https://github.com/open-mmlab/mmpose/pull/216))
- Improve README ([\#181](https://github.com/open-mmlab/mmpose/pull/181), [\#183](https://github.com/open-mmlab/mmpose/pull/183), [\#208](https://github.com/open-mmlab/mmpose/pull/208))
- Support return heatmaps and backbone features ([\#196](https://github.com/open-mmlab/mmpose/pull/196), [\#212](https://github.com/open-mmlab/mmpose/pull/212))
- Support different return formats of mmdetection models ([\#217](https://github.com/open-mmlab/mmpose/pull/217))

## v0.7.0 (30/9/2020)

**Highlights**

1. Support HMR for 3D human shape recovery.
1. Support WholeBody human pose estimation.
   - [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody)
1. Support more 2D hand keypoint estimation datasets.
   - [Frei-hand](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
   - [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html)
1. Add more popular backbones & enrich the [modelzoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html)
   - ShuffleNetv2
1. Support hand demo and whole-body demo.

**New Features**

- Support HMR for 3D human shape recovery ([\#157](https://github.com/open-mmlab/mmpose/pull/157), [\#160](https://github.com/open-mmlab/mmpose/pull/160), [\#161](https://github.com/open-mmlab/mmpose/pull/161), [\#162](https://github.com/open-mmlab/mmpose/pull/162))
- Support [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) dataset ([\#133](https://github.com/open-mmlab/mmpose/pull/133))
- Support [Frei-hand](https://lmb.informatik.uni-freiburg.de/projects/freihand/) dataset ([\#125](https://github.com/open-mmlab/mmpose/pull/125))
- Support [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html) dataset ([\#144](https://github.com/open-mmlab/mmpose/pull/144))
- Support H36M dataset ([\#159](https://github.com/open-mmlab/mmpose/pull/159))
- Support ShuffleNetv2 ([\#139](https://github.com/open-mmlab/mmpose/pull/139))
- Support saving best models based on key indicator ([\#127](https://github.com/open-mmlab/mmpose/pull/127))

**Bug Fixes**

- Fix typos in docs ([\#121](https://github.com/open-mmlab/mmpose/pull/121))
- Fix assertion ([\#142](https://github.com/open-mmlab/mmpose/pull/142))

**Improvements**

- Add tools to transform .mat format to .json format ([\#126](https://github.com/open-mmlab/mmpose/pull/126))
- Add hand demo ([\#115](https://github.com/open-mmlab/mmpose/pull/115))
- Add whole-body demo ([\#163](https://github.com/open-mmlab/mmpose/pull/163))
- Reuse mmcv utility function and update version files ([\#135](https://github.com/open-mmlab/mmpose/pull/135), [\#137](https://github.com/open-mmlab/mmpose/pull/137))
- Enrich the modelzoo ([\#147](https://github.com/open-mmlab/mmpose/pull/147), [\#169](https://github.com/open-mmlab/mmpose/pull/169))
- Improve docs ([\#174](https://github.com/open-mmlab/mmpose/pull/174), [\#175](https://github.com/open-mmlab/mmpose/pull/175), [\#178](https://github.com/open-mmlab/mmpose/pull/178))
- Improve README ([\#176](https://github.com/open-mmlab/mmpose/pull/176))
- Improve version.py ([\#173](https://github.com/open-mmlab/mmpose/pull/173))

## v0.6.0 (31/8/2020)

**Highlights**

1. Add more popular backbones & enrich the [modelzoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html)
   - ResNext
   - SEResNet
   - ResNetV1D
   - MobileNetv2
   - ShuffleNetv1
   - CPM (Convolutional Pose Machine)
1. Add more popular datasets:
   - [AIChallenger](https://arxiv.org/abs/1711.06475?context=cs.CV)
   - [MPII](http://human-pose.mpi-inf.mpg.de/)
   - [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)
   - [OCHuman](http://www.liruilong.cn/projects/pose2seg/index.html)
1. Support 2d hand keypoint estimation.
   - [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)
1. Support bottom-up inference.

**New Features**

- Support [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html) dataset ([\#52](https://github.com/open-mmlab/mmpose/pull/52))
- Support [MPII](http://human-pose.mpi-inf.mpg.de/) dataset ([\#55](https://github.com/open-mmlab/mmpose/pull/55))
- Support [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) dataset ([\#19](https://github.com/open-mmlab/mmpose/pull/19), [\#47](https://github.com/open-mmlab/mmpose/pull/47), [\#48](https://github.com/open-mmlab/mmpose/pull/48))
- Support [OCHuman](http://www.liruilong.cn/projects/pose2seg/index.html) dataset ([\#70](https://github.com/open-mmlab/mmpose/pull/70))
- Support [AIChallenger](https://arxiv.org/abs/1711.06475?context=cs.CV) dataset ([\#87](https://github.com/open-mmlab/mmpose/pull/87))
- Support multiple backbones ([\#26](https://github.com/open-mmlab/mmpose/pull/26))
- Support CPM model ([\#56](https://github.com/open-mmlab/mmpose/pull/56))

**Bug Fixes**

- Fix configs for MPII & MPII-TRB datasets ([\#93](https://github.com/open-mmlab/mmpose/pull/93))
- Fix the bug of missing `test_pipeline` in configs ([\#14](https://github.com/open-mmlab/mmpose/pull/14))
- Fix typos ([\#27](https://github.com/open-mmlab/mmpose/pull/27), [\#28](https://github.com/open-mmlab/mmpose/pull/28), [\#50](https://github.com/open-mmlab/mmpose/pull/50), [\#53](https://github.com/open-mmlab/mmpose/pull/53), [\#63](https://github.com/open-mmlab/mmpose/pull/63))

**Improvements**

- Update benchmark ([\#93](https://github.com/open-mmlab/mmpose/pull/93))
- Add Dockerfile ([\#44](https://github.com/open-mmlab/mmpose/pull/44))
- Improve unittest coverage and minor fix ([\#18](https://github.com/open-mmlab/mmpose/pull/18))
- Support CPUs for train/val/demo ([\#34](https://github.com/open-mmlab/mmpose/pull/34))
- Support bottom-up demo ([\#69](https://github.com/open-mmlab/mmpose/pull/69))
- Add tools to publish model ([\#62](https://github.com/open-mmlab/mmpose/pull/62))
- Enrich the modelzoo ([\#64](https://github.com/open-mmlab/mmpose/pull/64), [\#68](https://github.com/open-mmlab/mmpose/pull/68), [\#82](https://github.com/open-mmlab/mmpose/pull/82))

## v0.5.0 (21/7/2020)

**Highlights**

- MMPose is released.

**Main Features**

- Support both top-down and bottom-up pose estimation approaches.
- Achieve higher training efficiency and higher accuracy than other popular codebases (e.g. AlphaPose, HRNet)
- Support various backbone models: ResNet, HRNet, SCNet, Houglass and HigherHRNet.
