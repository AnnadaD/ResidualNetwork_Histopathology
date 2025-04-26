# ResidualNetwork_Histopthology

# Breast Cancer Detection Using Hybrid Deep Learning

Breast cancer is one of the most common cancers affecting women worldwide and remains a leading cause of mortality. Early and accurate detection is critical for improving survival rates, but the current standard—pathological image analysis—can be slow, labor-intensive, and dependent on highly specialized expertise. This challenge is even more pronounced in regions with limited medical resources and access to advanced diagnostic tools.
To bridge this gap, we introduce a deep learning-based solution designed to make breast cancer detection faster, more reliable, and accessible. At its heart is the **ResNet101** architecture, a powerful neural network known for extracting complex features from images. To take it a step further, we’ve incorporated a **Residual Convolutional Block Attention Module (CBAM)**. This attention mechanism helps the model focus on the most important parts of an image, both spatially and across its channels, significantly improving accuracy.



## Table of Contents

1. [Introduction](#introduction)
2. [Motivation and Contribution](#motivation-and-contribution)
3. [Materials and Method](#materials-and-method)
   - [Datasets and Pre-processing](#datasets-and-pre-processing)
   - [ResNet101](#resnet101)
   - [Convolutional Block Attention Module (CBAM)](#convolutional-block-attention-module-cbam)
   - [Proposed Hybrid Model Structure](#proposed-hybrid-model-structure)
4. [Experimental Results and Performance Analysis](#experimental-results-and-performance-analysis)
   - [Experimental Setup and Hyper-Parameter Details](#experimental-setup-and-hyper-parameter-details)
   - [Results and Analysis](#results-and-analysis)
   

---

## Introduction
Breast cancer is the most prevalent cancer worldwide, significantly impacting women's health and remaining a leading cause of mortality. While traditional pathological image analysis is highly effective, it is often slow, resource-intensive, and reliant on specialized expertise, limiting its accessibility in underserved regions.

This project presents an innovative hybrid deep learning approach to enhance breast cancer detection. By integrating ResNet101 with a Residual Convolutional Block Attention Module (CBAM), the model focuses on key spatial and channel-specific features in histopathological images. Combined with advanced preprocessing techniques, this solution achieves high diagnostic accuracy, making breast cancer detection faster, more accessible, and reliable.

---

## Motivation and Contribution

### Motivation
Traditional pathological analysis is resource-intensive and time-consuming. Automated methods leveraging deep learning can significantly enhance diagnostic speed and accuracy, particularly in under-resourced regions.

### Key Contributions
1. Integration of **ResNet101** with **CBAM** for enhanced spatial and channel-wise feature extraction.
2. A robust preprocessing pipeline including:
   - Geometric transformations (random flips and rotations)
   - CLAHE for contrast enhancement
   - Brightness and contrast adjustments.
3. State-of-the-art performance on **BreakHis** and **IDC** datasets.
4. Additional evaluation on the **MIAS** dataset to validate robustness across modalities.

---

## Materials and Method

### Datasets and Pre-processing
- **BreakHis:** 7,909 images at four magnifications (40x, 100x, 200x, 400x) classified as benign or malignant.
- **IDC:** 277,524 samples annotated as IDC-positive or IDC-negative.
- **MIAS:** 322 mammogram images categorized as normal, benign, or malignant.

### ResNet101
The ResNet101 architecture, pre-trained on ImageNet, serves as the backbone for hierarchical feature extraction. It utilizes residual connections to prevent vanishing gradients and enhance deep network training.

### Convolutional Block Attention Module (CBAM)
CBAM refines feature representations by applying **channel attention** and **spatial attention** sequentially, enabling the network to focus on relevant regions and features.
### Pipeline of the model
![pipeline](https://github.com/user-attachments/assets/f4cfeea7-9cb1-4c3c-8388-38ee109cea92)


### Hybrid Model Structure
A combination of ResNet101 with CBAM forms a robust hybrid model. Key features include:
- Fully connected layers with dropout and batch normalization for regularization.
- A sigmoid output layer for binary classification.
![resnet+cbam](https://github.com/user-attachments/assets/7ca6dc89-882a-4152-b6de-6ad4f54e6b1f)
---

## Experimental Results and Performance Analysis

### Experimental Setup and Hyper-Parameter Details
- Optimizer: Adam with a learning rate of 0.0001.
- Loss Function: Binary Cross-Entropy (BCE).
- Epochs: 100.
- Batch Size: 16.
- Image Resolution: 224 × 224 pixels.

### Results and Analysis
#### BreakHis Dataset
| Magnification | Accuracy (%) | Precision | Recall | F1-Score |
|---------------|--------------|-----------|--------|----------|
| 40x           | 99.75        | 99.78     | 99.72  | 99.75    |
| 100x          | 99.93        | 99.90     | 99.85  | 99.87    |
| 200x          | 99.87        | 99.70     | 99.88  | 99.89    |
| 400x          | 99.80        | 99.35     | 99.30  | 99.32    |

#### IDC Dataset
- **Accuracy:** 96.57%
- **Precision:** 94.03%
- **Recall:** 92.00%
- **F1-Score:** 93.01%

#### MIAS Dataset
- **Accuracy:** 99.03%
- **Precision:** 98.41%
- **Recall:** 99.56%
- **F1-Score:** 99.01%

# DOI: 
https://doi.org/10.1109/ISACC65211.2025.10969381

If you use our code please cite our paper as 
@INPROCEEDINGS{10969381,
author={Dash, Annada and Pramanik, Payel and Sarkar, Ram}, booktitle={2025 3rd International Conference on Intelligent Systems, Advanced Computing and Communication (ISACC)}, 
title={Attention-Enhanced Residual Network for Breast Cancer Detection in Histopathological Images}, year={2025}, volume={}, number={}, pages={697-703}, keywords={Deep learning;Image analysis;Attention mechanisms;Histopathology;Pipelines;Computer architecture;Feature extraction;Breast cancer;Data models;Residual neural networks;Breast cancer;Deep learning;Residual network;Attention module;Histopathology;CBAM;Medical image analysis},
doi={10.1109/ISACC65211.2025.10969381}}
