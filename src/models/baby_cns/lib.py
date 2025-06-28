
from .cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt
from .cns.utils.perception import CameraIntrinsic
import numpy as np
import json
import torch
 
class CnsLib:
    def __init__(self,device):
        self.device = device
        self.pipeline = self.create_pipeline(device)

    
    def create_pipeline(self):
        return CorrespondenceBasedPipeline(
            detector="AKAZE",
            ckpt_path="models/baby_cns/checkpoints/checkpoints/cns_state_dict.pth",
            intrinsic=CameraIntrinsic.default(),
            device=self.device or "cpu",
            ransac=True,
            vis=VisOpt.ALL,
        )
        
    def set_target(self, frame, dist_scale=1.0):
        """
        Set the target frame for the CNS pipeline.
        :param frame: The reference frame to set as target.
        :param dist_scale: Distance scale factor for the target.
        """
        self.pipeline.set_target(frame, dist_scale=dist_scale)
        
    def get_control_rate(self, frame):
        """
        Get the control rate for the given frame.
        :param frame: The current frame to process.
        :return: Velocity, data, and timing information.
        """
        vel, data_p, timing = self.pipeline.get_control_rate(frame)
        
        if data_p is not None:
            print(f"[SUCCESS] Pipeline execution successful")
            print(f"[INFO] Velocity output shape: {vel.shape if hasattr(vel, 'shape') else type(vel)}")
            print(f"[INFO] Velocity values: {vel}")
            print(f"[INFO] Timing information: {timing}")
        else:
            print("[ERROR] Pipeline returned None data - processing failed")
            print("This could indicate:")
            print("- Insufficient feature matches between images")
            print("- Images too different for correspondence")
            print("- Pipeline configuration issue")
        
        return vel, data_p, timing
    
    
    def save_results(self, results, output_file):
        """
        Save the results to a JSON file.
        :param results: The results to save.
        :param output_file: The path to the output file.
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Results saved to {output_file}")
       
               
    @staticmethod
    def convert_to_json_serializable(obj):
        """Convert numpy arrays and tensors to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: CnsLib.convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [CnsLib.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        # Handle PyTorch Geometric GraphData or similar objects
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'Data':
            # Only serialize public attributes (skip callables and private)
            return {k: CnsLib.convert_to_json_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_') and not callable(v)}
        else:
            return str(obj)  # fallback: convert to string
    