"""
ViT Feature Extractor Module
Gestisce l'estrazione di feature usando Vision Transformer (DINOv2)
"""

import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
import torch.nn.functional as F
from torchvision import transforms
import types
import math
from typing import Union, Tuple, List, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


class ViTExtractor:
    """ViT feature extractor for DINOv2"""
    
    def __init__(self, model_type: str = 'dinov2_vits14', stride: int = 2, device: str = None):
        self.model_type = model_type
        
        # Auto-detect device with environment variable support
        if device is None:
            import os
            cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"ViTExtractor using device: {self.device}")
        
        self.model = self.create_model(model_type)
        self.model = self.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        
        self.p = self.model.patch_embed.patch_size
        if isinstance(self.p, tuple):
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """Create ViT model"""
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        
        if 'dinov2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', model_type)
        elif 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """Fix position encoding for different strides"""
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            
            assert (w0 * h0 == npatch), f"Grid size mismatch: {h0}x{w0}={h0*w0} != {npatch}"
            
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False,
                recompute_scale_factor=False
            )
            
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """Patch ViT resolution by changing stride"""
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        
        if stride == patch_size:
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in stride]), \
               f'stride {stride} should divide patch_size {patch_size}'

        model.patch_embed.proj.stride = stride
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride), model
        )
        return model

    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray], 
                        load_size: Optional[int] = None) -> Tuple[torch.Tensor, Image.Image]:
        """Preprocess image for ViT"""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("Unsupported image type")

        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)

        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        prep_img = prep(pil_image)[None, ...].to(self.device)
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """Generate hook for feature extraction"""
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx])

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """Register hooks for feature extraction"""
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """Unregister hooks"""
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def extract_features(self, image: Union[str, Path, Image.Image, np.ndarray], 
                        layers: List[int] = [11], facet: str = 'key', 
                        load_size: Optional[int] = None) -> List[torch.Tensor]:
        """Extract features from ViT"""
        batch, _ = self.preprocess_image(image, load_size)
        
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        
        with torch.no_grad():
            _ = self.model(batch)
        
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        
        return self._feats

    def detect_vit_features(self, goal_image, current_image, num_pairs=10, dino_input_size=518):
        """Rileva feature usando ViT con chunked matching per gestire grandi risoluzioni"""
        # Estrai feature ViT da entrambe le immagini usando token features
        goal_feats = self.extract_features(
            goal_image, 
            load_size=dino_input_size, 
            facet='token'  # Usa token features invece di key/query/value
        )
        current_feats = self.extract_features(
            current_image, 
            load_size=dino_input_size,
            facet='token'
        )
        
        if not goal_feats or not current_feats:
            print("❌ Errore nell'estrazione delle feature ViT")
            return None, None
        
        # Prendi le feature dall'ultimo layer
        goal_feat = goal_feats[0]  # [1, num_patches, feature_dim]
        current_feat = current_feats[0]  # [1, num_patches, feature_dim]
        
        print(f"DEBUG: goal_feat shape: {goal_feat.shape}")
        print(f"DEBUG: current_feat shape: {current_feat.shape}")
        
        # Rimuovi il batch dimension
        if goal_feat.dim() == 3:
            goal_feat = goal_feat.squeeze(0)  # [num_patches, feature_dim]
            current_feat = current_feat.squeeze(0)  # [num_patches, feature_dim]
        
        # Rimuovi il token di classe [CLS] che è il primo token
        if goal_feat.shape[0] > 1:
            goal_feat = goal_feat[1:, :]  # [num_patches-1, feature_dim]
            current_feat = current_feat[1:, :]  # [num_patches-1, feature_dim]
        
        print(f"DEBUG: goal_feat final shape: {goal_feat.shape}")
        print(f"DEBUG: current_feat final shape: {current_feat.shape}")
        
        # Calcola chunk size dinamico in base al numero di patches
        num_patches = goal_feat.shape[0]
        if num_patches < 2000:
            chunk_size = num_patches  # Nessun chunking necessario
        elif num_patches < 5000:
            chunk_size = 1000
        elif num_patches < 10000:
            chunk_size = 500
        else:
            chunk_size = 250  # Chunk molto piccoli per risoluzioni estreme
        
        # GPU con 48GB VRAM - Usa chunked matching per evitare OOM
        print(f"🚀 Processing {goal_feat.shape[0]} patches with chunked matching...")
        
        # Verifica memoria GPU disponibile
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            free_memory = total_memory - allocated_memory
            print(f"💾 GPU Memory: {allocated_memory/1e9:.1f}GB used, {free_memory/1e9:.1f}GB free of {total_memory/1e9:.1f}GB total")
        
        # Normalizza le feature per similarità coseno
        goal_feat_norm = F.normalize(goal_feat, dim=-1)
        current_feat_norm = F.normalize(current_feat, dim=-1)
        
        # Chunked matching per gestire grandi quantità di patches
        num_goal_patches = goal_feat_norm.shape[0]
        num_current_patches = current_feat_norm.shape[0]
        
        print(f"📊 Goal patches: {num_goal_patches}, Current patches: {num_current_patches}")
        print(f"🔧 Using chunk size: {chunk_size}")
        
        # Arrays per memorizzare i migliori matches
        best_current_indices = torch.zeros(num_goal_patches, dtype=torch.long, device=self.device)
        best_similarities_gc = torch.zeros(num_goal_patches, device=self.device)
        best_goal_indices = torch.zeros(num_current_patches, dtype=torch.long, device=self.device)
        best_similarities_cg = torch.zeros(num_current_patches, device=self.device)
        
        # Goal -> Current matching (in chunks)
        print("🔄 Processing Goal -> Current matching...")
        for i, start_idx in enumerate(range(0, num_goal_patches, chunk_size)):
            end_idx = min(start_idx + chunk_size, num_goal_patches)
            goal_chunk = goal_feat_norm[start_idx:end_idx]
            
            if i % 10 == 0:  # Progress update ogni 10 chunks
                progress = (start_idx / num_goal_patches) * 100
                print(f"   Progress: {progress:.1f}% ({start_idx}/{num_goal_patches})")
            
            # Calcola similarità per questo chunk
            chunk_similarities = torch.mm(goal_chunk, current_feat_norm.t())
            
            # Trova i migliori match per questo chunk
            chunk_best_indices = torch.argmax(chunk_similarities, dim=1)
            chunk_best_similarities = torch.max(chunk_similarities, dim=1)[0]
            
            # Memorizza i risultati
            best_current_indices[start_idx:end_idx] = chunk_best_indices
            best_similarities_gc[start_idx:end_idx] = chunk_best_similarities
            
            # Libera memoria del chunk
            del chunk_similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Current -> Goal matching (in chunks)
        print("🔄 Processing Current -> Goal matching...")
        for i, start_idx in enumerate(range(0, num_current_patches, chunk_size)):
            end_idx = min(start_idx + chunk_size, num_current_patches)
            current_chunk = current_feat_norm[start_idx:end_idx]
            
            if i % 10 == 0:  # Progress update ogni 10 chunks
                progress = (start_idx / num_current_patches) * 100
                print(f"   Progress: {progress:.1f}% ({start_idx}/{num_current_patches})")
            
            # Calcola similarità per questo chunk
            chunk_similarities = torch.mm(current_chunk, goal_feat_norm.t())
            
            # Trova i migliori match per questo chunk
            chunk_best_indices = torch.argmax(chunk_similarities, dim=1)
            chunk_best_similarities = torch.max(chunk_similarities, dim=1)[0]
            
            # Memorizza i risultati
            best_goal_indices[start_idx:end_idx] = chunk_best_indices
            best_similarities_cg[start_idx:end_idx] = chunk_best_similarities
            
            # Libera memoria del chunk
            del chunk_similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("✅ Chunked matching completato, ricerca corrispondenze bidirezionali...")
        
        # Trova matches consistenti (corrispondenze bidirezionali)
        consistent_matches = []
        similarities = []
        
        for i in range(len(best_current_indices)):
            j = best_current_indices[i].item()
            if j < len(best_goal_indices) and best_goal_indices[j].item() == i:  # Match bidirezionale
                consistent_matches.append((i, j))
                similarities.append(best_similarities_gc[i].item())
        
        if len(consistent_matches) < 4:
            print(f"❌ Matches consistenti insufficienti: {len(consistent_matches)} < 4")
            return None, None
        
        # Ordina per similarità decrescente e prendi i migliori
        match_data = list(zip(consistent_matches, similarities))
        match_data.sort(key=lambda x: x[1], reverse=True)
        
        num_matches_to_use = min(num_pairs, len(match_data))
        best_matches = [match_data[i][0] for i in range(num_matches_to_use)]
        
        # Converti indici patch in coordinate pixel
        h_patches, w_patches = self.num_patches
        stride_h, stride_w = self.stride[0], self.stride[1]
        patch_size = self.p
        
        # Calcola il fattore di scala dalle dimensioni originali
        load_h, load_w = self.load_size
        original_h, original_w = 480, 640  # Dimensioni target
        scale_h = original_h / load_h
        scale_w = original_w / load_w
        
        goal_points = []
        current_points = []
        
        for goal_idx, current_idx in best_matches:
            # Converti indice patch lineare in coordinate 2D
            goal_patch_y = goal_idx // w_patches
            goal_patch_x = goal_idx % w_patches
            
            current_patch_y = current_idx // w_patches
            current_patch_x = current_idx % w_patches
            
            # Calcola centro del patch in coordinate pixel (nell'immagine ridimensionata)
            goal_y = (goal_patch_y * stride_h + patch_size // 2)
            goal_x = (goal_patch_x * stride_w + patch_size // 2)
            
            current_y = (current_patch_y * stride_h + patch_size // 2)
            current_x = (current_patch_x * stride_w + patch_size // 2)
            
            # Scala alle dimensioni originali
            goal_x *= scale_w
            goal_y *= scale_h
            current_x *= scale_w
            current_y *= scale_h
            
            goal_points.append([goal_x, goal_y])
            current_points.append([current_x, current_y])
        
        goal_points = np.array(goal_points)
        current_points = np.array(current_points)
        
        print(f"✅ ViT feature matching completato: {len(best_matches)} corrispondenze")
        avg_similarity = np.mean([match_data[i][1] for i in range(num_matches_to_use)])
        print(f"📊 Similarità media: {avg_similarity:.4f}")
        
        return goal_points, current_points