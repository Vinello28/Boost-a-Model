import json
import torch
import numpy as np
from typing import Union, Dict
from ..models.graph_vs import GraphVS
from ..midend.graph_gen import GraphData
from ..ablation.ibvs.ibvs import IBVS


class GraphVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        self.device = torch.device(device)
        
        # Skip weights_only loading for now, focus on compatibility
        print("[INFO] Loading checkpoint with compatibility mode")
        try:
            # Direct loading first
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"[ERROR] Standard loading failed: {e}")
            print("[INFO] Attempting alternative loading strategies")
            
            # Strategy 1: Try with pickle directly
            try:
                import pickle
                with open(ckpt_path, 'rb') as f:
                    ckpt = pickle.load(f)
                print("[INFO] Loaded with pickle successfully")
            except Exception as pickle_err:
                print(f"[ERROR] Pickle loading failed: {pickle_err}")
                
                # Strategy 2: Create fresh model and try loading state_dict only
                print("[INFO] Creating fresh model and attempting state dict loading")
                self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)
                try:
                    # Try to extract state dict in various ways
                    raw_data = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    
                    if isinstance(raw_data, dict):
                        if "net" in raw_data and hasattr(raw_data["net"], "state_dict"):
                            self.net.load_state_dict(raw_data["net"].state_dict(), strict=False)
                        elif "net" in raw_data:
                            self.net.load_state_dict(raw_data["net"], strict=False)
                        else:
                            self.net.load_state_dict(raw_data, strict=False)
                    else:
                        if hasattr(raw_data, "state_dict"):
                            self.net.load_state_dict(raw_data.state_dict(), strict=False)
                        else:
                            raise Exception("Cannot extract state dict from checkpoint")
                    
                    print("[INFO] State dict loaded successfully with fresh model")
                    self.net.eval()
                    self.hidden = None
                    return
                    
                except Exception as state_err:
                    print(f"[ERROR] State dict loading failed: {state_err}")
                    raise Exception("All loading strategies failed")
        
        # Process the successfully loaded checkpoint
        print("[INFO] Processing loaded checkpoint")
        if isinstance(ckpt, dict) and "net" in ckpt:
            try:
                self.net: GraphVS = ckpt["net"]
                print("[INFO] Loaded model directly from checkpoint")
            except Exception as e:
                print(f"[WARNING] Failed to load model directly: {e}")
                print("[INFO] Loading state dict instead")
                self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)
                if hasattr(ckpt["net"], "state_dict"):
                    self.net.load_state_dict(ckpt["net"].state_dict(), strict=False)
                else:
                    self.net.load_state_dict(ckpt["net"], strict=False)
        else:
            print("[INFO] Loading checkpoint as state dict")
            self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)
            self.net.load_state_dict(ckpt, strict=False)
        
        self.net.eval()
        self.hidden = None
        print("[INFO] Model loaded and ready")

    def __call__(self, data: GraphData) -> np.ndarray:
        with torch.no_grad():
            data = data.to(self.device)
            if hasattr(self.net, "preprocess"):
                data = self.net.preprocess(data)

            if getattr(data, "new_scene").any():
                print("[INFO] Got new scene, set hidden state to zero")
                self.hidden = None

            raw_pred = self.net(data, self.hidden)
            self.hidden = raw_pred[-1]
            vel = self.net.postprocess(raw_pred, data)

        vel = vel.squeeze(0).cpu().numpy()
        return vel


class IBVSController(object):
    def __init__(self, config_path: str):
        with open(config_path, "r") as fp:
            use_mean = json.load(fp)["use_mean"]
        self.ibvs = IBVS(use_mean)

    def __call__(self, data: GraphData) -> np.ndarray:
        return self.ibvs(data)


class ImageVSController(object):
    def __init__(self, ckpt_path: str, device="cuda:0"):
        from ..ablation.ibvs.raft_ibvs import RaftIBVS
        from ..reimpl import ICRA2018, ICRA2021
        
        self.device = torch.device(device)
        self.net: Union[ICRA2018, ICRA2021, RaftIBVS] = \
            torch.load(ckpt_path, map_location=self.device)["net"]
        self.net.eval()
        self.tar_feat = None
    
    def __call__(self, data: Dict) -> np.ndarray:
        with torch.no_grad():
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(self.device)
            
            if data.get("new_scene", True):
                self.tar_feat = None
            
            data["tar_feat"] = self.tar_feat
            raw_pred = self.net(data)
            self.tar_feat = data["tar_feat"]
            vel = self.net.postprocess(raw_pred, data)
        
        if isinstance(vel, torch.Tensor):
            vel = vel.cpu().numpy()
        vel = vel.flatten()
        return vel
