# Boost-a-Model

[![License](https://img.shields.io/github/license/Vinello28/Boost-a-Model.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

## Overview

### Introduction and State Of The Art

Visual Servoing is a control technique used in robotics that guides the movements of a robot based on visual information. Traditional approaches such as Position-Based Visual Servoing (PBVS) and Image-Based Visual Servoing (IBVS) suffer from limitations in adaptability and robustness to environmental variations. This project compares two innovative deep learning-based approaches.

### Graph Neural Network-Based Approach (CNS)

**Model Concept and Functioning:** The CNS (Correspondence-encoded Neural Servoing) method introduces a control strategy based on the explicit representation of visual correspondences between a current image and a reference image, modeled as a graph. In this graph, each node represents a keypoint detected in the images, while edges encode local relationships derived from spatial proximity or descriptor similarity.

**Model Architecture:** The CNS architecture is composed of four main modules: keypoint extraction and matching, graph construction, GNN encoder, and decoder. The core of the architecture is the Graph Convolutional Gated Recurrent Unit (GConvGRU), an extension of the classic GRU recurrent network, designed to handle the temporal evolution of information and adapted to work on structured data such as graphs.

**Model Fine-Tuning:** Fine-tuning was performed for a total of 50 epochs, with a batch size of 16 and the use of teacher forcing during the initial epochs. Optimization was carried out using the AdamW algorithm, with an initial learning rate of 5×10⁻⁴, later reduced to 1×10⁻⁴ to ensure more stable fine-tuning, and a weight decay of 1×10⁻⁴.

### Vision Transformer-Based Approach (ViT-VS)

**Model Concept and Functioning:** The second approach is based on the use of a Vision Transformer (ViT) for extracting semantic features from images. For this purpose, the DINOv2 architecture was adopted, pre-trained on a large-scale dataset of 142 million images.

**Model Architecture:** The ViT-VS architecture consists of a sequence of independent modules interacting in an IBVS-based visual control loop. The main modules include feature extraction via ViT, matching and selection of guide points, contextual aggregation, initial rotational compensation, classical IBVS control, and stabilization with EMA.

**Modifications Applied:** Several structural modifications were made to the ViT-VS architecture to decouple it from ROS and Gazebo, including the removal of the rotation module and adapting the input preprocessing to support video recordings from high-definition cameras.

### Main Results

Experimental results show that both approaches significantly outperform classical methods in terms of robustness and generalization. CNS excels in control accuracy and temporal handling, while ViT-VS stands out for implementation simplicity and immediate adaptability to different setups.

### Conclusions and Discussion

The comparative analysis between CNS and ViT-VS highlights a clear trade-off between computational complexity and operational performance. ViT-VS, particularly the ViTs14 model, offers an optimal compromise for real-time embedded applications, providing high accuracy with extremely low inference times. CNS, with its GNN-based architecture and keypoint detectors like AKAZE, shows reactive behavior in the presence of noise and perturbations, but its computational cost is a significant barrier for real-time use in embedded environments.

To further enhance overall performance and expand model applicability, several research and development directions are proposed, including model optimization and quantization for ViT, integration of pruning and distillation techniques for CNS, and the development of hybrid pipelines.


## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments & Results](#experiments--results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Modular training and evaluation pipeline for deep learning models.
- Support for multiple computer vision architectures, including transformers.
- Tools for dataset preprocessing and augmentation.
- Experiment tracking and reproducibility.
- Visualizations for model predictions and metrics.

