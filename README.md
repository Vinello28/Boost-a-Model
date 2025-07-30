<div align="center">

# 🚀 Boost-a-Model

### *Advanced Deep Learning for Visual Servoing in Robotics*

[![License](https://img.shields.io/github/license/Vinello28/Boost-a-Model.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge)](https://github.com/psf/black)
[![Contributions](https://img.shields.io/badge/contributions-welcome-orange?style=for-the-badge)](CONTRIBUTING.md)

---

*A cutting-edge comparison of Graph Neural Networks (CNS) and Vision Transformers (ViT-VS) for robust visual servoing in robotics applications.*

</div>

## 📋 Overview

> **Visual Servoing** is a control technique used in robotics that guides robot movements based on visual information. This project revolutionizes traditional approaches by comparing two state-of-the-art deep learning methodologies.

### 🎯 Problem Statement

Traditional Visual Servoing methods like **Position-Based Visual Servoing (PBVS)** and **Image-Based Visual Servoing (IBVS)** face significant limitations:
- ❌ Limited adaptability to environmental variations
- ❌ Poor robustness to noise and perturbations  
- ❌ Difficulty handling complex visual scenarios

### 💡 Our Benchmark Solution

We present a comprehensive comparison of **two innovative deep learning approaches** that overcome these limitations through advanced neural architectures.

---

## 🧠 Methodologies

### 🕸️ Graph Neural Network-Based Approach (CNS)

<details>
<summary><strong>🔍 Click to expand CNS details</strong></summary>

**🎨 Model Concept:** The CNS (Correspondence-encoded Neural Servoing) method introduces a revolutionary control strategy based on explicit visual correspondences between current and reference images, modeled as dynamic graphs.

**🏗️ Architecture Components:**
- **🔍 Keypoint Detection:** Advanced feature extraction and matching
- **📊 Graph Construction:** Spatial and descriptor-based relationship modeling
- **🧠 GNN Encoder:** Graph Convolutional Gated Recurrent Unit (GConvGRU)
- **⚡ Decoder:** Temporal information processing for control commands

**🎯 Training Configuration:**
- **Epochs:** 50 with teacher forcing
- **Batch Size:** 16
- **Optimizer:** AdamW (LR: 5×10⁻⁴ → 1×10⁻⁴)
- **Regularization:** Weight decay 1×10⁻⁴

</details>

### 👁️ Vision Transformer-Based Approach (ViT-VS)

<details>
<summary><strong>🔍 Click to expand ViT-VS details</strong></summary>

**🎨 Model Concept:** Leverages the power of Vision Transformers using **DINOv2** architecture, pre-trained on 142M images, for semantic feature extraction in visual control loops.

**🏗️ Architecture Pipeline:**
- **🔍 Feature Extraction:** ViT-based semantic understanding
- **🎯 Point Matching:** Intelligent guide point selection
- **📊 Contextual Aggregation:** Multi-scale feature integration
- **🔄 IBVS Control:** Classical control with modern features
- **⚖️ EMA Stabilization:** Exponential moving average smoothing

**🛠️ Key Modifications:**
- **🚫 ROS/Gazebo Decoupling:** Standalone implementation
- **📹 HD Camera Support:** Optimized for high-definition video input
- **⚡ Real-time Performance:** Optimized for embedded applications

</details>

---

## 📊 Results & Performance

### 🏆 Key Achievements

| Metric | CNS (GNN) | ViT-VS | Traditional Methods |
|--------|-----------|---------|-------------------|
| **🎯 Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **⚡ Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **🛡️ Robustness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **🔧 Simplicity** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 📈 Performance Highlights

- **🚀 Superior Performance:** Both approaches significantly outperform classical methods
- **🎯 CNS Strengths:** Exceptional control accuracy and temporal handling
- **⚡ ViT-VS Advantages:** Implementation simplicity and immediate adaptability
- **🔄 Trade-offs:** Clear balance between computational complexity and operational performance

### 🔍 Detailed Analysis

**🏅 ViT-VS (ViTs14)** emerges as the optimal choice for **real-time embedded applications**, offering:
- ✅ High accuracy with extremely low inference times
- ✅ Excellent compromise between performance and computational efficiency

**🧠 CNS with GNN Architecture** demonstrates:
- ✅ Reactive behavior under noise and perturbations  
- ✅ Superior temporal information handling
- ❌ Higher computational cost (barrier for real-time embedded use)

---

## 🎯 Conclusions & Future Directions

### 💡 Key Insights

The comparative analysis reveals a **fundamental trade-off** between computational complexity and operational performance:

- **🚀 ViT-VS Excellence:** Optimal for real-time applications with superior speed-accuracy balance
- **🔬 CNS Precision:** Advanced temporal modeling with higher computational requirements
- **📊 Hybrid Potential:** Combining both approaches could unlock even greater performance

### 🔮 Future Research Directions

<details>
<summary><strong>🛠️ Optimization Strategies</strong></summary>

- **⚡ Model Quantization:** Reduce ViT computational overhead
- **✂️ Pruning Techniques:** Streamline CNS architecture  
- **🧠 Knowledge Distillation:** Transfer learning between models
- **🔄 Hybrid Pipelines:** Combine GNN and Transformer strengths

</details>

---

## 📚 Table of Contents

- [🚀 Features](#-features)
- [🏗️ Project Structure](#️-project-structure)
- [⚙️ Installation](#️-installation)
- [🎮 Usage](#-usage)
- [🧪 Experiments & Results](#-experiments--results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 🚀 Features

<div align="center">

| Feature | Description |
|---------|-------------|
| 🔧 **Modular Pipeline** | Complete training and evaluation framework for deep learning models |
| 🏗️ **Multi-Architecture Support** | CNS (GNN) and ViT-VS (Transformer) implementations |
| 📊 **Advanced Preprocessing** | Comprehensive dataset tools and augmentation pipelines |
| 📈 **Experiment Tracking** | Built-in reproducibility and metrics visualization |
| 🐳 **Docker Support** | Containerized environment with CUDA acceleration |
| ⚡ **Real-time Inference** | Optimized for embedded and real-time applications |

</div>

---

## 🏗️ Project Structure

```
Boost-a-Model/
├── 📁 src/                    # Source code
│   ├── 🧠 models/            
│   │   ├── baby_cns/         # CNS (Graph Neural Network) implementation
│   │   └── vitvs/            # ViT-VS (Vision Transformer) implementation
│   ├── 🛠️ util/              # Utility functions and helpers
│   ├── ⚙️ config/            # Configuration files
│   └── 🎮 main.py            # Main execution script
├── 🐳 Dockerfile             # Container configuration
├── 🚀 run.sh                 # Docker run script
├── 📋 requirements.txt       # Python dependencies
└── 📖 README.md              # This file
```

---

## ⚙️ Installation

### 🐳 Docker Installation (Recommended)

**Prerequisites:**
- 🐳 Docker with NVIDIA runtime support
- 🎮 NVIDIA GPU with CUDA 11.8+ support

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Vinello28/Boost-a-Model.git
cd Boost-a-Model

# 2️⃣ Build and run the Docker container
chmod +x run.sh
./run.sh

# 3️⃣ Inside the container, set up the environment
source setup.fish
```

### 🐍 Local Installation

**Prerequisites:**
- 🐍 Python 3.12+
- 🚀 CUDA 11.8+ (for GPU acceleration)

```bash
# 1️⃣ Clone and navigate
git clone https://github.com/Vinello28/Boost-a-Model.git
cd Boost-a-Model/src

# 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3️⃣ Install PyTorch with CUDA support
pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 4️⃣ Install other dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### 🚀 Quick Start

```bash
# Run with default configuration
python src/main.py --method cns --input_video path/to/input.mp4 --goal_video path/to/reference.mp4

# Run ViT-VS approach
python src/main.py --method vitvs --input_video path/to/input.mp4 --goal_video path/to/reference.mp4
```

### ⚙️ Configuration

```bash
# Custom configuration
python src/main.py \
    --method cns \
    --input_video input.mp4 \
    --goal_video reference.mp4 \
    --output_dir results/ \
    --device cuda:0 \
    --batch_size 16
```

### 📊 Available Methods

| Method | Command | Description |
|--------|---------|-------------|
| **🕸️ CNS** | `--method cns` | Graph Neural Network approach |
| **👁️ ViT-VS** | `--method vitvs` | Vision Transformer approach |

---

## 🧪 Experiments & Results

### 📈 Running Benchmarks

```bash
# Run comprehensive benchmark
python src/main.py --benchmark --all_methods

# Generate performance report
python src/util/generate_report.py --results_dir results/
```

### 📊 Visualization

The framework automatically generates:
- 📈 **Performance Metrics:** Accuracy, speed, robustness analysis
- 🎥 **Visual Trajectories:** Control path visualization
- 📉 **Convergence Plots:** Training and validation curves
- 🔍 **Feature Maps:** Attention and activation visualizations

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🛠️ Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/YOUR_USERNAME/Boost-a-Model.git
cd Boost-a-Model

# Create feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m "✨ Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

### 📋 Contribution Guidelines

- ✅ Follow existing code style and conventions
- ✅ Add tests for new functionality
- ✅ Update documentation as needed
- ✅ Ensure all tests pass before submitting

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

### 🏆 Special Thanks

- **🧠 CNS Architecture:** Based on correspondence-encoded neural servoing research
- **👁️ DINOv2:** Meta AI's self-supervised vision transformer
- **🐍 PyTorch Community:** For the amazing deep learning framework
- **🚀 Open Source Contributors:** Making this project possible

### 📚 Citations

If you use this work in your research, please cite:

```bibtex
@misc{boost-a-model,
  title={Boost-a-Model: Deep Learning Approaches for Visual Servoing},
  author={Vinello28},
  year={2024},
  url={https://github.com/Vinello28/Boost-a-Model}
}
```

---

<div align="center">

### 🌟 Star this repo if you found it helpful!

[![GitHub stars](https://img.shields.io/github/stars/Vinello28/Boost-a-Model?style=social)](https://github.com/Vinello28/Boost-a-Model/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Vinello28/Boost-a-Model?style=social)](https://github.com/Vinello28/Boost-a-Model/network)

**Made with ❤️ for the robotics and computer vision community**

</div>

