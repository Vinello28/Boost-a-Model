#!/usr/bin/env python3
"""
Test script for CNS Image Processor

This script creates test images and demonstrates the CNS Image Processor functionality.
Use this to verify the installation and basic functionality.
"""

import cv2
import numpy as np
import os
import tempfile
from cns_image_processor import CNSImageProcessor


def create_test_images():
    """Create simple test images with features"""
    # Create a simple test pattern with features
    height, width = 480, 640
    
    # Goal image - checkerboard pattern
    goal_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create checkerboard
    square_size = 40
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                goal_image[i:i+square_size, j:j+square_size] = [255, 255, 255]
    
    # Add some distinctive features
    cv2.circle(goal_image, (160, 120), 30, (0, 0, 255), -1)  # Red circle
    cv2.circle(goal_image, (480, 120), 30, (0, 255, 0), -1)  # Green circle
    cv2.circle(goal_image, (320, 240), 30, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(goal_image, (250, 300), (390, 350), (255, 255, 0), -1)  # Yellow rectangle
    
    # Current image - similar but slightly transformed
    current_image = goal_image.copy()
    
    # Apply slight rotation and translation to simulate camera movement
    center = (width // 2, height // 2)
    rotation_angle = 5  # degrees
    scale = 0.95
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    
    # Add translation
    M[0, 2] += 20  # x translation
    M[1, 2] += 10  # y translation
    
    # Apply transformation
    current_image = cv2.warpAffine(current_image, M, (width, height), 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(50, 50, 50))
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, current_image.shape).astype(np.uint8)
    current_image = cv2.add(current_image, noise)
    
    return goal_image, current_image


def test_cns_processor():
    """Test the CNS Image Processor with generated test images"""
    print("="*60)
    print("TESTING CNS IMAGE PROCESSOR")
    print("="*60)
    
    try:
        # Create test images
        print("[INFO] Creating test images...")
        goal_image, current_image = create_test_images()
        
        # Save test images to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            goal_path = os.path.join(temp_dir, "goal_test.jpg")
            current_path = os.path.join(temp_dir, "current_test.jpg")
            
            cv2.imwrite(goal_path, goal_image)
            cv2.imwrite(current_path, current_image)
            
            print(f"[INFO] Test images saved to:")
            print(f"  Goal: {goal_path}")
            print(f"  Current: {current_path}")
            
            # Initialize processor (try different settings if CUDA fails)
            print("\n[INFO] Initializing CNS processor...")
            
            try:
                processor = CNSImageProcessor(
                    detector="SIFT",
                    device="cuda:0",
                    distance_prior=0.5,
                    visualize=True
                )
                print("[INFO] Using CUDA device")
            except Exception as e:
                print(f"[WARNING] CUDA failed: {e}")
                print("[INFO] Falling back to CPU")
                processor = CNSImageProcessor(
                    detector="SIFT", 
                    device="cpu",
                    distance_prior=0.5,
                    visualize=True
                )
            
            # Process images
            print("\n[INFO] Processing test images...")
            result = processor.process_images(
                goal_image_path=goal_path,
                current_image_path=current_path,
                save_visualization=True,
                output_dir="test_output"
            )
            
            # Display results
            if result.success:
                print("\n[SUCCESS] Test completed successfully!")
                print("\nTest Results Summary:")
                print(f"  - Detector: {result.detector_name}")
                print(f"  - Keypoints in goal: {len(result.keypoints_goal)}")
                print(f"  - Keypoints in current: {len(result.keypoints_current)}")
                print(f"  - Matches found: {result.num_matches}")
                print(f"  - Velocity magnitude: {np.linalg.norm(result.velocity):.4f}")
                print(f"  - Processing time: {result.timing.get('total_time', 0)*1000:.2f} ms")
                
                # Show visualization
                if result.correspondence_image is not None:
                    print("\n[INFO] Displaying result visualization...")
                    print("Press any key to close the window.")
                    cv2.imshow("CNS Test Result", result.correspondence_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                return True
                
            else:
                print(f"\n[ERROR] Test failed: {result.error_message}")
                return False
                
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if all required dependencies are available"""
    print("="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    dependencies = {
        "OpenCV": "cv2",
        "NumPy": "numpy", 
        "PyTorch": "torch",
        "PyTorch Geometric": "torch_geometric"
    }
    
    missing = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}: Available")
        except ImportError:
            print(f"✗ {name}: Missing")
            missing.append(name)
    
    # Check CNS modules
    try:
        from cns.benchmark.pipeline import CorrespondenceBasedPipeline
        print("✓ CNS modules: Available")
    except ImportError as e:
        print(f"✗ CNS modules: Missing - {e}")
        missing.append("CNS")
    
    # Check for model checkpoint
    checkpoint_paths = ["checkpoints/cns.pth", "checkpoints/cns_state_dict.pth"]
    checkpoint_found = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✓ Model checkpoint: {path}")
            checkpoint_found = True
            break
    
    if not checkpoint_found:
        print("✗ Model checkpoint: Not found")
        print("  Expected locations:", checkpoint_paths)
        missing.append("Model checkpoint")
    
    if missing:
        print(f"\n[WARNING] Missing dependencies: {', '.join(missing)}")
        print("Please install missing dependencies before using CNS Image Processor")
        return False
    else:
        print("\n[SUCCESS] All dependencies available!")
        return True


def main():
    """Main test function"""
    print("CNS IMAGE PROCESSOR - TESTING SCRIPT")
    print("="*60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n[ERROR] Cannot proceed due to missing dependencies")
        return
    
    print("\n")
    
    # Run test
    success = test_cns_processor()
    
    if success:
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The CNS Image Processor is working correctly.")
        print("You can now use it with your own images.")
    else:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        print("Please check the error messages above and ensure all dependencies are installed.")


if __name__ == "__main__":
    main()
