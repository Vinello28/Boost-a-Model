import torch
import numpy as np
import cv2
import os
import json
from cns.utils.perception import CameraIntrinsic
from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt


def set_seed(seed=2023):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def convert_to_json_serializable(obj):
    """Convert numpy arrays and tensors to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def load_image(image_path):
    """Load image from path and return as BGR uint8 array"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return img


def run_cns_with_external_images():
    """
    Run CNS pipeline with external images.
    Uses comandovitruviano.jpeg as goal image and curr3.jpeg as current image.
    """
    # Set seed for reproducibility
    set_seed()
    
    # Initialize CNS pipeline with same configuration as demo
    pipeline = CorrespondenceBasedPipeline(
        detector="AKAZE",
        # detector="SuperGlue:0123",  # Alternative detector
        ckpt_path="checkpoints/cns_state_dict.pth",
        intrinsic=CameraIntrinsic.default(),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        ransac=True,
        vis=VisOpt.MATCH | VisOpt.GRAPH
    )
    
    # Define image paths
    goal_image_path = "dataset_small/comandovitruviano.jpeg"
    current_image_path = "dataset_small/curr3.jpeg"
    
    try:
        # Load images
        print(f"[INFO] Loading goal image: {goal_image_path}")
        goal_img = load_image(goal_image_path)
        
        print(f"[INFO] Loading current image: {current_image_path}")
        current_img = load_image(current_image_path)
        
        print(f"[INFO] Goal image shape: {goal_img.shape}")
        print(f"[INFO] Current image shape: {current_img.shape}")
        
        # Set the target image (goal)
        # Assuming a default distance scale (you may need to adjust this)
        dist_scale = 1.0
        pipeline.set_target(goal_img, dist_scale=dist_scale)
        
        # Process current image to get control rate
        print("[INFO] Processing images with CNS pipeline...")
        vel, data, timing = pipeline.get_control_rate(current_img)
        
        # Print results
        print("\n" + "="*60)
        print("CNS PIPELINE RESULTS")
        print("="*60)
        
        if data is not None:
            print(f"[SUCCESS] Pipeline execution successful")
            print(f"[INFO] Velocity output shape: {vel.shape if hasattr(vel, 'shape') else type(vel)}")
            print(f"[INFO] Velocity values: {vel}")
            print(f"[INFO] Timing information: {timing}")
            
            # Print data information if available
            if hasattr(data, 'keys'):
                print(f"[INFO] Data keys: {list(data.keys())}")
            
            # Save results if needed
            results = {
                "velocity": convert_to_json_serializable(vel),
                "data": convert_to_json_serializable(data),
                "timing": convert_to_json_serializable(timing),
                "goal_image_path": goal_image_path,
                "current_image_path": current_image_path,
                "goal_image_shape": list(goal_img.shape),
                "current_image_shape": list(current_img.shape),
                "device_used": "cuda:0" if torch.cuda.is_available() else "cpu",
                "detector": "AKAZE"
            }
            
            # Save results to JSON file
            output_file = "cns_external_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[INFO] Results saved to {output_file}")
            
        else:
            print("[ERROR] Pipeline returned None data - processing failed")
            print("This could indicate:")
            print("- Insufficient feature matches between images")
            print("- Images too different for correspondence")
            print("- Pipeline configuration issue")
        
        print("="*60)
        
        return vel, data, timing
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None, None, None


def analyze_correspondence(goal_image_path="dataset_small/comandovitruviano.jpeg", 
                          current_image_path="dataset_small/curr3.jpeg"):
    """
    Analyze correspondence between two images using CNS frontend only.
    This function focuses on feature detection and matching.
    """
    set_seed()
    
    # Initialize pipeline
    pipeline = CorrespondenceBasedPipeline(
        detector="AKAZE",
        ckpt_path="checkpoints/cns_state_dict.pth",
        intrinsic=CameraIntrinsic.default(),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        ransac=True,
        vis=VisOpt.MATCH | VisOpt.GRAPH
    )
    
    try:
        # Load images
        goal_img = load_image(goal_image_path)
        current_img = load_image(current_image_path)
        
        print(f"[INFO] Analyzing correspondence between:")
        print(f"  Goal: {goal_image_path}")
        print(f"  Current: {current_image_path}")
        
        # Set target
        pipeline.set_target(goal_img, dist_scale=1.0)
        
        # Get frontend analysis
        vel, data, timing = pipeline.get_control_rate(current_img)
        
        if data is not None:
            print(f"[SUCCESS] Found correspondences between images")
            if hasattr(data, 'keys'):
                for key in data.keys():
                    print(f"  {key}: {type(data[key])}")
        else:
            print(f"[WARNING] No correspondences found between images")
            
        return data
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return None


if __name__ == "__main__":
    print("CNS External Images Interface")
    print("============================")
    
    # Run main processing
    vel, data, timing = run_cns_with_external_images()
    
    print("\n" + "-"*60)
    print("Additional Analysis")
    print("-"*60)
    
    # Run correspondence analysis
    correspondence_data = analyze_correspondence()
    
    print("\n[INFO] Processing complete!")
