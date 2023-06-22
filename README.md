# Discrepancy Scaling
Discrepancy Scaling is a fast method for unsupervised anomaly detection and localization. This repository will house the official implementation of the method, presented in the paper "Discrepancy Scaling for Fast Unsupervised
Anomaly Localization". This PyTorch implementation of Discrepancy Scaling is built on top of the [official implementation](https://github.com/gdwang08/STFPM) of Student-Teacher Feature Pyramid Matching.

## The state of this repository
This repository now contains a working but very basic implementation of Discrepancy Scaling. We are refactoring the code for better legibility and usability. We are also adding documentation.

## How to get started
All necessary functions and classes are in `d_utils.py`. The notebook demonstrates how to use them. You can obtain parameters for pre-trained student-teacher models at the [STFPM](https://github.com/gdwang08/STFPM) repository. Have fun!
