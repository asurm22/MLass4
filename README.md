# MLass4
Overview

This project implements a Convolutional Neural Network (CNN) pipeline for the ICML Facial Expression Recognition Challenge. It extends a baseline CNN with class balancing, advanced data augmentations (MixUp, CutMix, Random Erasing), Focal Loss, OneCycleLR scheduling, and mixed-precision training to improve validation and test accuracies.

Features

ImprovedCNN architecture with deeper convolutional blocks and adaptive pooling

Class balancing via WeightedRandomSampler

Augmentations:

MixUp

CutMix

RandomErasing

Standard flips, rotations, crops, color jitter

Loss function: Focal Loss to emphasize hard examples

Learning rate scheduling: OneCycleLR for dynamic LR adjustment

Mixed-precision training: torch.cuda.amp for speed and stability

WandB integration for experiment tracking, metrics, and confusion matrix logging
