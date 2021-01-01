# 2D Face Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [300W](#300w-dataset) \[ [Homepage](https://ibug.doc.ic.ac.uk/resources/300-W/) \]
- [WFLW](#wflw-dataset) \[ [Homepage](https://wywu.github.io/projects/LAB/WFLW.html) \]
- [AFLW](#aflw-dataset) \[ [Homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) \]
- [COFW](#cofw-dataset) \[ [Homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/) \]

## 300W Dataset

For 300W data, please download images from [300W Dataset](https://ibug.doc.ic.ac.uk/resources/300-W/).
Please download the annotation files from [300w_annotations](https://download.openmmlab.com/mmpose/datasets/300w_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── 300w
        |── annotations
        |   |── face_landmarks_300w_train.json
        |   |── face_landmarks_300w_valid.json
        |   |── face_landmarks_300w_valid_common.json
        |   |── face_landmarks_300w_valid_challenge.json
        |   |── face_landmarks_300w_test.json
        `── images
            |── afw
            |   |── 1051618982_1.jpg
            |   |── 111076519_1.jpg
            |    ...
            |── helen
            |   |── trainset
            |   |   |── 100032540_1.jpg
            |   |   |── 100040721_1.jpg
            |   |    ...
            |   |── testset
            |   |   |── 296814969_3.jpg
            |   |   |── 2968560214_1.jpg
            |   |    ...
            |── ibug
            |   |── image_003_1.jpg
            |   |── image_004_1.jpg
            |    ...
            |── lfpw
            |   |── trainset
            |   |   |── image_0001.png
            |   |   |── image_0002.png
            |   |    ...
            |   |── testset
            |   |   |── image_0001.png
            |   |   |── image_0002.png
            |   |    ...
            `── Test
                |── 01_Indoor
                |   |── indoor_001.png
                |   |── indoor_002.png
                |    ...
                `── 02_Outdoor
                    |── outdoor_001.png
                    |── outdoor_002.png
                     ...
```

## WFLW Dataset

For WFLW data, please download images from [WFLW Dataset](https://wywu.github.io/projects/LAB/WFLW.html).
Please download the annotation files from [wflw_annotations](https://download.openmmlab.com/mmpose/datasets/wflw_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── 300w
        |── annotations
        |   |── face_landmarks_wflw_train.json
        |   |── face_landmarks_wflw_test.json
        |   |── face_landmarks_wflw_test_blur.json
        |   |── face_landmarks_wflw_test_occlusion.json
        |   |── face_landmarks_wflw_test_expression.json
        |   |── face_landmarks_wflw_test_occlusion.json
        |   |── face_landmarks_wflw_test_largepose.json
        |   |── face_landmarks_wflw_test_occlusion.json
        |   |── face_landmarks_wflw_test_illumination.json
        |   |── face_landmarks_wflw_test_makeup.json
        |
        `── images
            |── 0--Parade
            |   |── 0_Parade_marchingband_1_1015.jpg
            |   |── 0_Parade_marchingband_1_1031.jpg
            |    ...
            |── 1--Handshaking
            |   |── 1_Handshaking_Handshaking_1_105.jpg
            |   |── 1_Handshaking_Handshaking_1_107.jpg
            |    ...
            ...
```

## AFLW Dataset

For AFLW data, please download images from [AFLW Dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/).
Please download the annotation files from [aflw_annotations](https://download.openmmlab.com/mmpose/datasets/aflw_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── aflw
        |── annotations
        |   |── face_landmarks_aflw_train.json
        |   |── face_landmarks_aflw_test_frontal.json
        |   |── face_landmarks_aflw_test.json
        `── images
            |── flickr
                |── 0
                |   |── image00002.jpg
                |   |── image00013.jpg
                |    ...
                |── 2
                |   |── image00004.jpg
                |   |── image00006.jpg
                |    ...
                `── 3
                    |── image00032.jpg
                    |── image00035.jpg
                     ...
```

## COFW Dataset

For COFW data, please download from [COFW Dataset (Color Images)](http://www.vision.caltech.edu/xpburgos/ICCV13/).
Move `COFW_train_color.mat` and `COFW_test_color.mat` to `data/cofw/` and make them look like:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── cofw
        |── COFW_train_color.mat
        |── COFW_test_color.mat
```

Run the following script under `$MMPOSE`

`python tools/dataset/parse_cofw_dataset.py`

And you will get

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── cofw
        |── COFW_train_color.mat
        |── COFW_test_color.mat
        |── annotations
        |   |── cofw_train.json
        |   |── cofw_test.json
        |── images
            |── 000001.jpg
            |── 000002.jpg
```