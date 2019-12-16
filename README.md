# Enhancing Feature Fusion for Human Pose Estimation
A new method to fuse high-level features and low-level features in human pose estimation

## Introduction
This code refers to SimpleBaseline: https://github.com/microsoft/human-pose-estimation.pytorch. we use Semantic Embedding Block (SEB) and Global Convolutional Network (GCN) blocks to bridge the gap between low-level and high-level features. Experiments on MPII and LSP human pose estimation datasets demonstrate that efficient feature fusion can significantly improve the performance.

## Results on MPII val
| Method | Input | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean |
|---|---|---|---|---|---|---|---|---|---|
| SimpleBaseline_ResNet50| 256x256 | 96.35 | 95.33 | 88.99 | 83.18 | 88.42 | 83.96 | 79.59 | 88.53 |
| ours | 256x256 | 96.73 | 95.35 | 89.50 | 83.73 | 88.23 | 84.43 | 79.92 | 88.82 |
| SimpleBaseline_ResNet50| 384x384 | 96.66 | 95.75 | 89.79 | 84.61 | 88.52 | 84.67 | 79.29 | 89.07 |
| ours | 384x384 | 96.67 | 95.75 | 90.05 | 85.58 | 88.85 | 84.73 | 79.74| 89.35 |



## Environment
python >= 3.6 \
pytorch >= 1.0.0

## Quick start
1. Download the dataset and pretrained model, you can follow the an official pytorch implementation of SimpleBaseline.
2. Training the model:

```
python pose_estimation/train.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```
3. valid the model:

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```


### Future work
look forward to multi-scale feature fusion structures.
