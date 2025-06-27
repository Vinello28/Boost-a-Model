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
        
        try:
            # Add GraphVS to safe globals for weights_only loading
            torch.serialization.add_safe_globals([GraphVS])
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except (TypeError, Exception) as e:
            print(f"[WARNING] weights_only loading failed: {e}")
            print("[INFO] Falling back to standard loading")
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint: {e}")
                print("[INFO] Attempting to load with strict=False")
                # Create model and try partial loading
                self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    self.net.load_state_dict(ckpt, strict=False)
                except Exception as load_err:
                    print(f"[ERROR] All loading methods failed: {load_err}")
                    raise load_err
                self.net.eval()
                self.hidden = None
                return
        
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
