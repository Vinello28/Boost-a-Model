# Boost-a-Model

Computer Vision and Deep Learning project focused on training and evaluating various models for Visual Servoing applications.

## About The Project

This repository contains the implementation of two primary visual servoing methods:

- **ViT-VS**: A Visual Servoing approach based on the Vision Transformer (ViT), which leverages self-supervised learning to extract robust features for precise control.
- **CNS**: A Classical Visual Servoing method that uses traditional feature detectors like SIFT, AKAZE, and ORB for correspondence-based control.

The project is designed to be modular and extensible, allowing for the integration of new models and techniques.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.8+
- Poetry
- Git

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/Boost-a-Model.git
   cd Boost-a-Model
   ```

2. **Install dependencies**:

   ```bash
   poetry install
   ```

## Usage

### ViT-VS

To run the ViT-VS method, use the following command:

```bash
poetry run python src/main.py --method test-vit-vs --reference /path/to/goal.mp4 --input /path/to/stream.mp4
```

### CNS

To run the CNS method, use the following command:

```bash
poetry run python src/main.py --method cns --reference /path/to/goal.mp4 --input /path/to/stream.mp4
```

## Authors

- Began Bajrami
- Mattia Mandorlini
- Gabriele Vianello
