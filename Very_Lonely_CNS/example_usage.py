#!/usr/bin/env python3
"""
Example usage of the CNS Image Processor

This script demonstrates how to use the CNS Image Processor both programmatically
and through command line interface.
"""

import os
import numpy as np
from cns_image_processor import CNSImageProcessor, CNSResult


def example_programmatic_usage():
    """Example of using CNS Image Processor programmatically"""
    print("="*60)
    print("EXAMPLE: Programmatic Usage")
    print("="*60)
    
    # Initialize the processor
    processor = CNSImageProcessor(
        detector="SIFT",  # Can also use AKAZE, ORB, SuperGlue, etc.
        device="cuda:0",  # Use "cpu" if CUDA is not available
        distance_prior=0.5,  # Estimated distance to scene in meters
        visualize=True
    )
    
    # Process images (replace with actual image paths)
    goal_image = "path/to/your/goal_image.jpg"
    current_image = "path/to/your/current_image.jpg"
    
    # Check if example images exist (this is just for demo)
    if not (os.path.exists(goal_image) and os.path.exists(current_image)):
        print("[INFO] Please provide actual image paths in the example")
        print(f"[INFO] Expected goal image: {goal_image}")
        print(f"[INFO] Expected current image: {current_image}")
        return
    
    # Process the images
    result = processor.process_images(
        goal_image_path=goal_image,
        current_image_path=current_image,
        save_visualization=True,
        output_dir="example_output"
    )
    
    # Use the results
    if result.success:
        print(f"\n[SUCCESS] Processing completed!")
        print(f"Velocity command: {result.velocity}")
        print(f"Number of matches: {result.num_matches}")
        
        # You could now send the velocity command to your robot
        # robot.move_camera(result.velocity)
        
    else:
        print(f"\n[ERROR] Processing failed: {result.error_message}")


def example_command_line_usage():
    """Show command line usage examples"""
    print("\n" + "="*60)
    print("EXAMPLE: Command Line Usage")
    print("="*60)
    
    examples = [
        "# Basic usage with default settings:",
        "python cns_image_processor.py --goal goal.jpg --current current.jpg",
        "",
        "# Using different detector:",
        "python cns_image_processor.py --goal goal.jpg --current current.jpg --detector AKAZE",
        "",
        "# Using CPU instead of GPU:",
        "python cns_image_processor.py --goal goal.jpg --current current.jpg --device cpu",
        "",
        "# Custom distance prior and output directory:",
        "python cns_image_processor.py --goal goal.jpg --current current.jpg --distance 0.3 --output my_results",
        "",
        "# Using SuperGlue detector (if available):",
        "python cns_image_processor.py --goal goal.jpg --current current.jpg --detector SuperGlue",
        "",
        "# Don't save visualization files:",
        "python cns_image_processor.py --goal goal.jpg --current current.jpg --no-save"
    ]
    
    for example in examples:
        print(example)


def explain_output():
    """Explain what the tool outputs"""
    print("\n" + "="*60)
    print("OUTPUT EXPLANATION")
    print("="*60)
    
    explanation = """
The CNS Image Processor provides the following outputs:

1. VELOCITY VECTOR:
   - 6-DoF camera velocity in camera coordinate frame
   - Format: [vx, vy, vz, wx, wy, wz]
   - Linear velocities (vx, vy, vz) in m/s
   - Angular velocities (wx, wy, wz) in rad/s
   - This can be directly sent to a robot controller

2. KEYPOINT INFORMATION:
   - Detected keypoints in goal image
   - Detected keypoints in current image  
   - Matched keypoint pairs
   - Number of valid matches

3. VISUALIZATIONS:
   - Side-by-side image comparison
   - Keypoints marked on both images
   - Lines connecting matched keypoints
   - Color-coded matches for easy interpretation

4. TIMING INFORMATION:
   - Frontend processing time (feature detection/matching)
   - Midend processing time (graph construction)
   - Backend processing time (neural network inference)
   - Total processing time

5. SAVED FILES:
   - correspondence_visualization.jpg: Visual comparison image
   - analysis_results.json: Complete analysis in JSON format
   - keypoints_and_matches.npz: Keypoints and matches as NumPy arrays

INTERPRETING RESULTS:
- More matches generally mean better servo performance
- Velocity magnitude indicates how far the camera needs to move
- Low processing times indicate real-time capability
- Failed matches may indicate poor lighting, blur, or insufficient features
"""
    
    print(explanation)


def integration_tips():
    """Tips for integrating with robot systems"""
    print("\n" + "="*60)
    print("ROBOT INTEGRATION TIPS")
    print("="*60)
    
    tips = """
1. COORDINATE FRAMES:
   - CNS outputs velocity in camera coordinate frame
   - You may need to transform to robot base frame
   - Consider camera mounting orientation

2. VELOCITY SCALING:
   - CNS outputs normalized velocities
   - Scale by appropriate gains for your robot
   - Start with small gains and increase gradually

3. STOPPING CRITERIA:
   - Use the number of matches to assess convergence
   - Stop when velocity magnitude becomes small
   - Consider using the built-in stop policies

4. ERROR HANDLING:
   - Check result.success before using velocity
   - Handle cases with no detected matches
   - Have fallback strategies for poor lighting

5. REAL-TIME CONSIDERATIONS:
   - Monitor timing information
   - Consider using faster detectors (ORB vs SIFT)
   - GPU acceleration recommended for real-time use

6. PARAMETER TUNING:
   - Adjust distance_prior based on your setup
   - Try different detectors for your environment
   - Tune gains based on your robot dynamics
"""
    
    print(tips)


def main():
    """Run all examples and explanations"""
    example_programmatic_usage()
    example_command_line_usage()
    explain_output()
    integration_tips()
    
    print("\n" + "="*60)
    print("READY TO USE!")
    print("="*60)
    print("The CNS Image Processor is ready for use.")
    print("Modify the image paths in example_programmatic_usage() to test with real images.")


if __name__ == "__main__":
    main()
