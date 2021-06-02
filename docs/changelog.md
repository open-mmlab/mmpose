# Changelog

## v0.15.0 (02/06/2021)

**Highlights**

1. Support 3d video pose estimation (VideoPose3D).
1. Support 3d hand pose estimation (InterNet).
1. Improve presentation of modelzoo.

**New Features**

- Support "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image" (ECCV‘20) ([\#624](https://github.com/open-mmlab/mmpose/pull/624))
- Support "3D human pose estimation in video with temporal convolutions and semi-supervised training" (CVPR'19)  ([\#602](https://github.com/open-mmlab/mmpose/pull/602),[\#681](https://github.com/open-mmlab/mmpose/pull/681))
- Support 3d pose estimation demo ([\#653](https://github.com/open-mmlab/mmpose/pull/653),[\#670](https://github.com/open-mmlab/mmpose/pull/670))
- Support bottom-up whole-body pose estimation ([\#689](https://github.com/open-mmlab/mmpose/pull/689))
- Support mmcli ([\#634](https://github.com/open-mmlab/mmpose/pull/634))

**Bug Fixes**

- Fix opencv compatibility ([\#635](https://github.com/open-mmlab/mmpose/pull/635))
- Fix demo with UDP ([\#637](https://github.com/open-mmlab/mmpose/pull/637))
- Fix bottom-up model onnx convertion ([\#680](https://github.com/open-mmlab/mmpose/pull/680))
- Fix `GPU_IDS` in distributed training ([\#668](https://github.com/open-mmlab/mmpose/pull/668))
- Fix MANIFEST.in ([\#641](https://github.com/open-mmlab/mmpose/pull/641),[\#657](https://github.com/open-mmlab/mmpose/pull/657))
- Fix docs ([\#643](https://github.com/open-mmlab/mmpose/pull/643),[\#684](https://github.com/open-mmlab/mmpose/pull/684),[\#688](https://github.com/open-mmlab/mmpose/pull/688),[\#690](https://github.com/open-mmlab/mmpose/pull/690),[\#692](https://github.com/open-mmlab/mmpose/pull/692))

**Breaking Changes**

- Reorganize configs by tasks, algorithms, datasets, and techniques ([\#647](https://github.com/open-mmlab/mmpose/pull/647))
- Rename heads and detectors ([\#667](https://github.com/open-mmlab/mmpose/pull/667))

**Improvements**

- Add `radius` and `thickness` parameters in visualization ([\#638](https://github.com/open-mmlab/mmpose/pull/638))
- Add `trans_prob` parameter in `TopDownRandomTranslation` ([\#650](https://github.com/open-mmlab/mmpose/pull/650))
- Switch to `MMCV MODEL_REGISTRY` ([\#669](https://github.com/open-mmlab/mmpose/pull/669))
- Update dependencies ([\#674](https://github.com/open-mmlab/mmpose/pull/674),[\#676](https://github.com/open-mmlab/mmpose/pull/676))

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
- Update github CI ([\#562](https://github.com/open-mmlab/mmpose/pull/562),[\#564](https://github.com/open-mmlab/mmpose/pull/564))
- Update Readme ([\#578](https://github.com/open-mmlab/mmpose/pull/578),[\#579](https://github.com/open-mmlab/mmpose/pull/579),[\#580](https://github.com/open-mmlab/mmpose/pull/580),[\#592](https://github.com/open-mmlab/mmpose/pull/592),[\#599](https://github.com/open-mmlab/mmpose/pull/599),[\#600](https://github.com/open-mmlab/mmpose/pull/600),[\#607](https://github.com/open-mmlab/mmpose/pull/607))
- Update Faq ([\#587](https://github.com/open-mmlab/mmpose/pull/587),[\#610](https://github.com/open-mmlab/mmpose/pull/610))

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
