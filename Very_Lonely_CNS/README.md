# Very Lonely CNS - Standalone CNS Image Processor

🎯 **Standalone version of CNS Image Processor**

This is a self-contained version of the CNS (Correspondence Encoded Neural Image Servo Policy) Image Processor that can be used independently from the main CNS repository.

## What's Inside

This folder contains everything you need to run CNS image analysis:

```
Very_Lonely_CNS/
├── cns_image_processor.py      # Main processor (CLI + API)
├── example_usage.py            # Usage examples and documentation
├── test_cns_processor.py       # Testing script with synthetic images
├── pipeline.json               # Default configuration
├── requirements.txt            # Python dependencies
├── setup_venv.bat             # Windows setup script (batch)
├── setup_venv.ps1             # Windows setup script (PowerShell)
├── activate_venv.bat          # Quick activation script
├── docker_setup.sh            # Linux/Mac setup script (bash)
├── README.md                  # This file
├── checkpoints/                # Pre-trained CNS models
│   ├── cns.pth
│   ├── cns_state_dict.pth
│   └── ibvs_config.json
└── cns/                        # Complete CNS module
    ├── benchmark/              # Pipeline and evaluation tools
    ├── frontend/               # Feature detection and matching
    ├── midend/                 # Graph construction
    ├── models/                 # Neural network models
    ├── utils/                  # Utility functions
    └── ...
```

## Quick Start

### 🚀 Easy Setup (Recommended)

**For Windows Users:**
```batch
# Double-click or run:
setup_venv.bat

# Or use PowerShell:
.\setup_venv.ps1
```

**For Linux/Mac Users:**
```bash
chmod +x docker_setup.sh
./docker_setup.sh
```

This will:
- ✅ Create a Python virtual environment
- 📦 Install all required dependencies
- 🎮 Test GPU availability
- 📁 Create necessary directories (input/, results/, logs/)

### 🔄 Daily Usage

After initial setup, activate the environment:

**Windows:**
```batch
activate_venv.bat
# or manually: venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 1. Manual Installation (Alternative)

**Option A: Automatic (Recommended)**
```bash
python install_pytorch.py  # Smart installer - detects CUDA and installs appropriate version
```

**Option B: Manual**
```bash
pip install -r requirements.txt  # Installs default PyTorch (CUDA if available, CPU otherwise)
```

**Option C: Specific PyTorch Version**
```bash
# For CPU only
pip install torch>=1.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch>=1.12.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install torch-geometric opencv-python numpy scipy matplotlib tqdm scikit-image
```

### 2. Test Installation

```bash
python test_cns_processor.py
```

### 3. Process Your Images

```bash
# Command line usage
python cns_image_processor.py --goal your_goal_image.jpg --current your_current_image.jpg

# Programmatic usage
python -c "
from cns_image_processor import CNSImageProcessor
processor = CNSImageProcessor()
result = processor.process_images('goal.jpg', 'current.jpg')
print(f'Velocity: {result.velocity}')
"
```

## What You Get

- **6-DoF Camera Velocity**: Ready for robot control
- **Keypoint Analysis**: Detected features and matches
- **Visual Feedback**: Images with keypoints and correspondences
- **Performance Metrics**: Timing and quality information
- **Saved Results**: JSON, images, and NumPy arrays

## Features

### 🚀 **Multiple Detectors**
- SIFT (high quality)
- AKAZE (balanced)
- ORB (fast)
- SuperGlue (best quality, requires setup)

### 💻 **Flexible Usage**
- Command line interface
- Python API
- Batch processing
- Real-time capable

### 📊 **Rich Output**
- Velocity vectors for robot control
- Keypoint visualizations
- Timing analysis
- Quality metrics

### 🔧 **Configurable**
- Device selection (CPU/GPU)
- Distance priors
- Detector parameters
- Output formats

## Usage Examples

### Basic Command Line

```bash
# Process two images
python cns_image_processor.py --goal goal.jpg --current current.jpg

# Use different detector
python cns_image_processor.py --goal goal.jpg --current current.jpg --detector AKAZE

# Use CPU instead of GPU
python cns_image_processor.py --goal goal.jpg --current current.jpg --device cpu

# Custom distance prior
python cns_image_processor.py --goal goal.jpg --current current.jpg --distance 0.3
```

### Programmatic Usage

```python
from cns_image_processor import CNSImageProcessor

# Initialize processor
processor = CNSImageProcessor(
    detector="SIFT",
    device="cuda:0",
    distance_prior=0.5
)

# Process images
result = processor.process_images("goal.jpg", "current.jpg")

if result.success:
    # Get velocity for robot control
    velocity = result.velocity  # [vx, vy, vz, wx, wy, wz]
    
    # Check quality
    if result.num_matches > 20:
        print("High quality matches!")
        # Send to robot
        # robot.move_camera(velocity)
    else:
        print("Low quality matches, be careful!")
        
    # Print analysis
    print(f"Detector: {result.detector_name}")
    print(f"Matches: {result.num_matches}")
    print(f"Processing time: {result.timing['total_time']*1000:.1f}ms")
else:
    print(f"Processing failed: {result.error_message}")
```

### Batch Processing

```python
import os
from cns_image_processor import CNSImageProcessor

processor = CNSImageProcessor()
goal_image = "reference.jpg"

# Process multiple current images
for image_file in os.listdir("images/"):
    if image_file.endswith(('.jpg', '.png')):
        current_image = os.path.join("images", image_file)
        result = processor.process_images(goal_image, current_image)
        
        if result.success:
            print(f"{image_file}: {result.num_matches} matches, "
                  f"velocity magnitude: {np.linalg.norm(result.velocity):.3f}")
```

## Output Files

When processing completes, you'll find:

- **correspondence_visualization.jpg**: Side-by-side comparison with keypoint matches
- **analysis_results.json**: Complete analysis in JSON format
- **keypoints_and_matches.npz**: NumPy arrays with raw data

## Robot Integration

The velocity output is ready for robot control:

```python
result = processor.process_images("goal.jpg", "current.jpg")

if result.success:
    # 6-DoF velocity in camera frame
    linear_vel = result.velocity[:3]   # [vx, vy, vz] in m/s
    angular_vel = result.velocity[3:]  # [wx, wy, wz] in rad/s
    
    # Scale for your robot
    linear_vel *= linear_gain
    angular_vel *= angular_gain
    
    # Send to robot controller
    robot.set_camera_velocity(linear_vel, angular_vel)
```

## Performance Tips

- **GPU**: Use `--device cuda:0` for faster processing
- **Real-time**: Use ORB detector for fastest processing
- **Quality**: Use SIFT or SuperGlue for best results
- **Distance**: Underestimate distance prior if uncertain

## Troubleshooting

### No Matches Found
- Check image quality and lighting
- Try different detectors
- Ensure sufficient texture in images

### Slow Processing
- Use `--device cpu` if GPU issues
- Try faster detectors (ORB, AKAZE)
- Check available memory

### Import Errors
- Run `pip install -r requirements.txt`
- Check Python environment
- Verify PyTorch installation

## What Makes This "Very Lonely"?

This version is completely self-contained and doesn't depend on the parent CNS repository structure. You can:

- Copy this folder anywhere
- Run it independently
- Share it as a standalone tool
- Use it in your own projects

Perfect for when CNS just wants to analyze images by itself! 🤖

## Support

For detailed documentation, see `example_usage.py` and run `test_cns_processor.py` to verify your installation.

For issues related to the main CNS system, refer to the original CNS repository.
