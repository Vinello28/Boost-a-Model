#!/usr/bin/env python3
"""
CNS Image Processor - A user-friendly interface for CNS image servo analysis

This script allows you to use the CNS (Correspondence Encoded Neural Image Servo Policy) 
system with goal and current images as input, providing comprehensive output including:
- Velocity vectors (6-DoF camera velocity)
- Detected keypoints and matches
- Visual comparison of images with keypoints and correspondences
- Timing information
- Graph structure visualization

Usage:
    python cns_image_processor.py --goal path/to/goal_image.jpg --current path/to/current_image.jpg
    
Or use it as a module:
    from cns_image_processor import CNSImageProcessor
    processor = CNSImageProcessor()
    result = processor.process_images("goal.jpg", "current.jpg")
"""

import cv2
import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

# CNS imports
from cns.utils.perception import CameraIntrinsic
from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt
from cns.frontend.utils import plot_corr


@dataclass
class CNSResult:
    """Container for CNS processing results"""
    velocity: np.ndarray  # 6-DoF camera velocity [vx, vy, vz, wx, wy, wz]
    keypoints_goal: np.ndarray  # Goal image keypoints (N, 2)
    keypoints_current: np.ndarray  # Current image keypoints (M, 2)
    matches: np.ndarray  # Keypoint matches (K, 2)
    num_matches: int  # Number of valid matches
    timing: Dict[str, float]  # Timing information
    correspondence_image: np.ndarray  # Visualization image with keypoints and matches
    detector_name: str  # Name of the detector used
    success: bool  # Whether processing was successful
    error_message: Optional[str] = None


class CNSImageProcessor:
    """Main class for processing images with CNS"""
    
    def __init__(self, 
                 config_path: str = "pipeline.json",
                 detector: str = "SIFT",
                 device: str = "cuda:0",
                 checkpoint_path: str = "checkpoints/cns.pth",
                 distance_prior: float = 0.5,
                 visualize: bool = True,
                 intrinsic: Optional[CameraIntrinsic] = None):
        """
        Initialize the CNS Image Processor
        
        Args:
            config_path: Path to pipeline configuration JSON file
            detector: Feature detector to use (SIFT, AKAZE, ORB, SuperGlue, etc.)
            device: Device for computation ("cuda:0" or "cpu")
            checkpoint_path: Path to CNS model checkpoint
            distance_prior: Prior estimate of distance to scene center (meters)
            visualize: Whether to generate visualizations
            intrinsic: Camera intrinsic parameters (uses default if None)
        """
        self.distance_prior = distance_prior
        self.visualize = visualize
        
        # Try to load from config file first, then use parameters
        if os.path.exists(config_path):
            try:
                self.pipeline = CorrespondenceBasedPipeline.from_file(config_path)
                print(f"[INFO] Loaded pipeline from {config_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load from {config_path}: {e}")
                print("[INFO] Using provided parameters instead")
                self._create_pipeline_from_params(detector, device, checkpoint_path, intrinsic)
        else:
            print(f"[INFO] Config file {config_path} not found, using provided parameters")
            self._create_pipeline_from_params(detector, device, checkpoint_path, intrinsic)
            
    def _create_pipeline_from_params(self, detector, device, checkpoint_path, intrinsic):
        """Create pipeline from individual parameters"""
        if intrinsic is None:
            intrinsic = CameraIntrinsic.default()
            
        vis_opt = VisOpt.NO  # We'll handle visualization manually for better control
        
        self.pipeline = CorrespondenceBasedPipeline(
            detector=detector,
            ckpt_path=checkpoint_path,
            intrinsic=intrinsic,
            device=device,
            ransac=True,
            vis=vis_opt
        )
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image in BGR format as required by CNS"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        return image
        
    def process_images(self, 
                      goal_image_path: str, 
                      current_image_path: str,
                      save_visualization: bool = True,
                      output_dir: str = "output") -> CNSResult:
        """
        Process goal and current images to get CNS analysis
        
        Args:
            goal_image_path: Path to goal/target image
            current_image_path: Path to current image  
            save_visualization: Whether to save visualization images
            output_dir: Directory to save outputs
            
        Returns:
            CNSResult object containing all analysis results
        """
        try:
            # Load images
            goal_image = self.load_image(goal_image_path)
            current_image = self.load_image(current_image_path)
            
            print(f"[INFO] Loaded goal image: {goal_image.shape}")
            print(f"[INFO] Loaded current image: {current_image.shape}")
            
            # Set the target (goal) image
            success = self.pipeline.set_target(goal_image, dist_scale=self.distance_prior)
            if not success:
                return CNSResult(
                    velocity=np.zeros(6),
                    keypoints_goal=np.array([]),
                    keypoints_current=np.array([]),
                    matches=np.array([]),
                    num_matches=0,
                    timing={},
                    correspondence_image=np.zeros((100, 100, 3), dtype=np.uint8),
                    detector_name="Unknown",
                    success=False,
                    error_message="Failed to set target image"
                )
            
            # Get control rate and analysis
            velocity, data, timing = self.pipeline.get_control_rate(current_image)
            
            if data is None:
                return CNSResult(
                    velocity=velocity,
                    keypoints_goal=np.array([]),
                    keypoints_current=np.array([]),
                    matches=np.array([]),
                    num_matches=0,
                    timing=timing,
                    correspondence_image=self._create_no_matches_image(goal_image, current_image),
                    detector_name="Unknown",
                    success=False,
                    error_message="No keypoint correspondences found"
                )
            
            # Extract correspondence information
            frontend_data = self.pipeline.frontend.corr
            
            # Create visualization
            correspondence_image = plot_corr(frontend_data, show_keypoints=True)
            
            # Prepare result
            result = CNSResult(
                velocity=velocity,
                keypoints_goal=frontend_data.tar_pos,
                keypoints_current=frontend_data.cur_pos,
                matches=frontend_data.match,
                num_matches=np.sum(frontend_data.valid_mask),
                timing=timing,
                correspondence_image=correspondence_image,
                detector_name=frontend_data.detector_name,
                success=True
            )
            
            # Print results
            self._print_results(result)
            
            # Save outputs if requested
            if save_visualization:
                self._save_outputs(result, goal_image_path, current_image_path, output_dir)
                
            return result
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            return CNSResult(
                velocity=np.zeros(6),
                keypoints_goal=np.array([]),
                keypoints_current=np.array([]),
                matches=np.array([]),
                num_matches=0,
                timing={},
                correspondence_image=np.zeros((100, 100, 3), dtype=np.uint8),
                detector_name="Unknown",
                success=False,
                error_message=str(e)
            )
    
    def _create_no_matches_image(self, goal_image: np.ndarray, current_image: np.ndarray) -> np.ndarray:
        """Create a side-by-side image when no matches are found"""
        H0, W0 = goal_image.shape[:2]
        H1, W1 = current_image.shape[:2]
        H, W = max(H0, H1), W0 + W1 + 10
        
        out = np.ones((H, W, 3), dtype=np.uint8) * 255
        out[:H0, :W0] = goal_image
        out[:H1, W0+10:] = current_image
        
        # Add text
        cv2.putText(out, "NO MATCHES FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        cv2.putText(out, "Goal Image", (10, H0-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        cv2.putText(out, "Current Image", (W0+20, H1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return out
    
    def _print_results(self, result: CNSResult):
        """Print analysis results to console"""
        print("\n" + "="*80)
        print("CNS IMAGE PROCESSING RESULTS")
        print("="*80)
        
        if not result.success:
            print(f"[ERROR] Processing failed: {result.error_message}")
            return
            
        print(f"Detector: {result.detector_name}")
        print(f"Goal image keypoints: {len(result.keypoints_goal)}")
        print(f"Current image keypoints: {len(result.keypoints_current)}")
        print(f"Valid matches: {result.num_matches}")
        
        print(f"\nVelocity Command (camera frame):")
        print(f"  Linear velocity  [vx, vy, vz]: [{result.velocity[0]:.4f}, {result.velocity[1]:.4f}, {result.velocity[2]:.4f}] m/s")
        print(f"  Angular velocity [wx, wy, wz]: [{result.velocity[3]:.4f}, {result.velocity[4]:.4f}, {result.velocity[5]:.4f}] rad/s")
        print(f"  Velocity magnitude: {np.linalg.norm(result.velocity):.4f}")
        
        print(f"\nTiming Information:")
        for key, value in result.timing.items():
            print(f"  {key}: {value*1000:.2f} ms")
            
        print("="*80)
    
    def _save_outputs(self, 
                     result: CNSResult, 
                     goal_path: str, 
                     current_path: str, 
                     output_dir: str):
        """Save visualization and analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save correspondence visualization
        cv2.imwrite(os.path.join(output_dir, "correspondence_visualization.jpg"), 
                   result.correspondence_image)
        
        # Save analysis results as JSON
        analysis_data = {
            "goal_image_path": goal_path,
            "current_image_path": current_path,
            "detector": result.detector_name,
            "num_goal_keypoints": len(result.keypoints_goal),
            "num_current_keypoints": len(result.keypoints_current),
            "num_matches": result.num_matches,
            "velocity": {
                "linear": [float(x) for x in result.velocity[:3]],
                "angular": [float(x) for x in result.velocity[3:]],
                "magnitude": float(np.linalg.norm(result.velocity))
            },
            "timing": result.timing,
            "success": result.success,
            "error_message": result.error_message
        }
        
        with open(os.path.join(output_dir, "analysis_results.json"), 'w') as f:
            json.dump(analysis_data, f, indent=2)
            
        # Save keypoints as numpy arrays
        np.savez(os.path.join(output_dir, "keypoints_and_matches.npz"),
                goal_keypoints=result.keypoints_goal,
                current_keypoints=result.keypoints_current,
                matches=result.matches,
                velocity=result.velocity)
        
        print(f"\n[INFO] Results saved to: {output_dir}")
        print(f"  - correspondence_visualization.jpg")
        print(f"  - analysis_results.json") 
        print(f"  - keypoints_and_matches.npz")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="CNS Image Processing Tool")
    parser.add_argument("--goal", required=True, help="Path to goal/target image")
    parser.add_argument("--current", required=True, help="Path to current image")
    parser.add_argument("--config", default="pipeline.json", help="Pipeline config file")
    parser.add_argument("--detector", default="SIFT", help="Feature detector (SIFT, AKAZE, ORB, etc.)")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--checkpoint", default="checkpoints/cns.pth", help="Model checkpoint path")
    parser.add_argument("--distance", type=float, default=0.5, help="Distance prior (meters)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save visualization files")
    
    args = parser.parse_args()
    
    # Create processor
    processor = CNSImageProcessor(
        config_path=args.config,
        detector=args.detector,
        device=args.device,
        checkpoint_path=args.checkpoint,
        distance_prior=args.distance
    )
    
    # Process images
    result = processor.process_images(
        args.goal, 
        args.current,
        save_visualization=not args.no_save,
        output_dir=args.output
    )
    
    # Show visualization if successful
    if result.success and result.correspondence_image is not None:
        print("\n[INFO] Press any key to close the visualization window...")
        cv2.imshow("CNS Correspondence Analysis", result.correspondence_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
