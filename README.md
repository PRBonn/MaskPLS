# Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving

This repository contains the implementation of our [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf).

## Installation

* Install this package by running in the root directory of this repo:

```
pip3 install -U -e .
```

* Install the packages in [requirements.txt](requirements.txt).


## Data preparation

### SemanticKITTI
Download the [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#overview) dataset inside the directory `data/kitti/`. The directory structure should look like this:
```
./
└── data/
    └── kitti
        └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   └── labels/ 
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```

### NuScenes
We use [nuscenes2kitti](https://github.com/PRBonn/nuscenes2kitti) to convert the nuScenes format into the SemanticKITTI format and store it in `data/nuscenes/`.

In the scripts, use the `--nuscenes` flag to train or evaluate using this dataset.

## Pretrained models

* [SemanticKITTI](https://www.ipb.uni-bonn.de/html/projects/mask_based_panoptic_segmentation/mask_pls_kitti.ckpt)

* [NuScenes](https://www.ipb.uni-bonn.de/html/projects/mask_based_panoptic_segmentation/mask_pls_nuscenes.ckpt)

## Reproducing results
```
python3 scripts/evaluate_model.py --w [path_to_model]
```

## Training

```
python3 scripts/train_model.py

```

## Citation
```bibtex
@article{marcuzzi2023ral,
  author = {R. Marcuzzi and L. Nunes and L. Wiesmann and J. Behley and C. Stachniss},
  title = {{Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving}},
  journal = ral,
  volume = {8},
  number = {2},
  pages = {1141--1148},
  year = 2023,
  doi = {10.1109/LRA.2023.3236568},
  url = {https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf},
}
```
## Licence
Copyright 2023, Rodrigo Marcuzzi, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

This project is free software made available under the MIT License. For details see the LICENSE file
