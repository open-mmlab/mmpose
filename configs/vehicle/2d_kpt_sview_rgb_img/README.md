# 2D Body Vehicle Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- Images
  - [CARFUSION](#carfusion) \[ [Homepage](http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html) \]

## CARFUSION

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://www.cs.cmu.edu/~ILIM/publications/PDFs/RVN-CVPR18.pdf">CARFUSION (CVPR'2018)</a></summary>

```bibtex
@InProceedings{Reddy_2018_CVPR,
author = {Dinesh Reddy, N. and Vo, Minh and Narasimhan, Srinivasa G.},
title = {CarFusion: Combining Point Tracking and Part Detection for Dynamic 3D Reconstruction of Vehicles},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

</details>
CARFUSION provides manual annotations of 14 semantic keypoints
for 100,000 car instances (sedan, suv, bus, and truck) from 53,000 images
captured from 18 moving cameras at Multiple intersections in Pittsburgh, PA.
To download the data you need to fill the form [Access Form](https://forms.gle/FCUcbt3jD1hB6ja57)
and convert it to coco format using [CARFUSIONTOCOCO](https://github.com/dineshreddy91/carfusion_to_coco) :

Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── coco
        │-- annotations
        │   │-- carfusion_train.json
        │   |-- carfusion_test.json
        │-- images
        │   │-- carfusion
            	│-- train
            	    │-- car_butler1
            	     	│-- images_jpg
            	     	    │-- 11_00201.jpg
            	     	    │-- 11_00241.jpg
            	     	    │-- 11_00281.jpg
            	│-- test
            	    │-- car_penn1
            	     	│-- images_jpg
            	     	    │-- 10_0001.jpg
            	     	    │-- 10_0041.jpg
            	     	    │-- 10_0081.jpg



```
