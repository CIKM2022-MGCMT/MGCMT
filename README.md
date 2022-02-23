# MSCMT
The code implementation of this paper refers to [UMT](https://github.com/jefferyYu/UMT)

## Requirements

python>=3.6

torch>=1.7.1

pytorch-crf>=0.7.2

## Pre-trained Models
Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth)
Download the pre-trained Mask-RCNN via this link (https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)

## Datasets
Download twitter images from [UMT](https://github.com/jefferyYu/UMT/)

## Training for MSCMT
This is the training code of tuning parameters on the dev set, and testing on the test set:
```python

sh run_mtmner_crf.sh
```
