# SKFlow: Learning Optical Flow with Super Kernels

This repository is the official implementation of our paper:

[SKFlow: Learning Optical Flow with Super Kernels](https://arxiv.org/pdf/2205.14623v2.pdf)

Shangkun Sun, Yuanqi Chen, Yu Zhu, Guodong Guo, Ge Li

NeurIPS 2022

## Requirements
The code is tested on PyTorch 1.10.0, Python 3.9.7.
To install requirements:

```
pip install -r requirements.txt
```

## Data Preparation
SKFlow uses the following datasets for training and evaluation:
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)

Datasets are suggested to be organized as follows:
```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```


## Training

To train the model(s) in the paper, run this command:

```
sh scripts/train.sh
```

## Evaluation

To evaluate our model (e.g. on Sintel), run:

```
sh scripts/infer.sh
```

## Pre-trained Models

Pre-trained models could be downloaded here:

- [Pretrained Model for Sintel](https://drive.google.com/file/d/1F6Ag7MPG8QrP3RSXAzSSv42_RYyZx5P2/view?usp=sharing)
- [Pretrained Model for KITTI](https://drive.google.com/file/d/1IQL8edRo-DugTFV3UpL-mN0e4k2nzLOC/view?usp=sharing)
- [Pretrained Model on C+T](https://drive.google.com/file/d/18NuBPV4hjivnPsPHPTEbjdiACfqg30Al/view?usp=sharing)


## Acknowledgement
Parts of code are adapted from the following repositories. We thank the authors for their great contribution to the community:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
- [ptflops](https://github.com/sovrasov/flops-counter.pytorch)



