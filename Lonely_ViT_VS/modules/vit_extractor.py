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
import gc
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
                        load_size: Optional[int] = None, smart_crop: bool = True) -> Tuple[torch.Tensor, Image.Image]:
        """Preprocess image for ViT with patch size compatibility and smart cropping"""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("Unsupported image type")

        # Get original dimensions
        orig_w, orig_h = pil_image.size
        
        # 🎯 SMART CROPPING: Remove empty/background areas
        if smart_crop:
            pil_image = self._smart_crop_object(pil_image)
            print(f"🎯 Smart cropping applied: {orig_w}x{orig_h} → {pil_image.size[0]}x{pil_image.size[1]}")
        
        # CRITICAL: Ensure dimensions are multiples of patch_size (14)
        patch_size = 14
        
        if load_size is not None:
            # Resize and then ensure patch compatibility
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
            w, h = pil_image.size
        else:
            w, h = pil_image.size
        
        # Calculate new dimensions that are multiples of patch_size
        new_w = ((w + patch_size - 1) // patch_size) * patch_size  # Round up
        new_h = ((h + patch_size - 1) // patch_size) * patch_size  # Round up
        
        # If dimensions changed, resize to patch-compatible size
        if new_w != w or new_h != h:
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            print(f"🔧 Resized image from {w}x{h} to {new_w}x{new_h} for patch compatibility")

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
                        load_size: Optional[int] = None, smart_crop: bool = True) -> List[torch.Tensor]:
        """Extract features from ViT"""
        batch, _ = self.preprocess_image(image, load_size, smart_crop=smart_crop)
        
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        
        try:
            with torch.no_grad():
                _ = self.model(batch)
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA OOM in extract_features: {e}")
            # Pulizia memoria
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise
        finally:
            self._unregister_hooks()
            # Pulizia memoria dopo estrazione
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        
        return self._feats

    def detect_vit_features(self, goal_image, current_image, num_pairs=10, dino_input_size=518):
        """Rileva feature usando ViT con chunked matching per gestire grandi risoluzioni"""
        # Estrai feature ViT da entrambe le immagini usando token features
        # 🎯 SMART CROPPING: Applica solo alla goal image per concentrarsi sull'oggetto
        goal_feats = self.extract_features(
            goal_image, 
            load_size=dino_input_size, 
            facet='token',  # Usa token features invece di key/query/value
            smart_crop=True  # Crop automatico dell'oggetto nella goal image
        )
        current_feats = self.extract_features(
            current_image, 
            load_size=dino_input_size,
            facet='token',
            smart_crop=False  # Non croppare la current image (mantieni tutto il contesto)
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
        try:
            for i, start_idx in enumerate(range(0, num_goal_patches, chunk_size)):
                end_idx = min(start_idx + chunk_size, num_goal_patches)
                goal_chunk = goal_feat_norm[start_idx:end_idx]
                
                if i % 10 == 0:  # Progress update ogni 10 chunks
                    progress = (start_idx / num_goal_patches) * 100
                    print(f"   Progress: {progress:.1f}% ({start_idx}/{num_goal_patches})")
                
                try:
                    # Calcola similarità per questo chunk
                    chunk_similarities = torch.mm(goal_chunk, current_feat_norm.t())
                    
                    # Trova i migliori match per questo chunk
                    chunk_best_indices = torch.argmax(chunk_similarities, dim=1)
                    chunk_best_similarities = torch.max(chunk_similarities, dim=1)[0]
                    
                    # Memorizza i risultati
                    best_current_indices[start_idx:end_idx] = chunk_best_indices
                    best_similarities_gc[start_idx:end_idx] = chunk_best_similarities
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ CUDA OOM nel chunk {i}, provo con chunk size ridotto")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Prova con chunk size dimezzato
                    reduced_chunk_size = chunk_size // 2
                    if reduced_chunk_size < 10:
                        raise RuntimeError("Impossibile processare anche con chunk molto piccoli")
                    
                    # Riprocessa questo chunk con dimensione ridotta
                    for sub_start in range(start_idx, end_idx, reduced_chunk_size):
                        sub_end = min(sub_start + reduced_chunk_size, end_idx)
                        sub_chunk = goal_feat_norm[sub_start:sub_end]
                        
                        sub_similarities = torch.mm(sub_chunk, current_feat_norm.t())
                        sub_best_indices = torch.argmax(sub_similarities, dim=1)
                        sub_best_similarities = torch.max(sub_similarities, dim=1)[0]
                        
                        best_current_indices[sub_start:sub_end] = sub_best_indices
                        best_similarities_gc[sub_start:sub_end] = sub_best_similarities
                        
                        del sub_similarities
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Libera memoria del chunk
                if 'chunk_similarities' in locals():
                    del chunk_similarities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Errore durante Goal->Current matching: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return None, None
        
        # Current -> Goal matching (in chunks)
        print("🔄 Processing Current -> Goal matching...")
        try:
            for i, start_idx in enumerate(range(0, num_current_patches, chunk_size)):
                end_idx = min(start_idx + chunk_size, num_current_patches)
                current_chunk = current_feat_norm[start_idx:end_idx]
                
                if i % 10 == 0:  # Progress update ogni 10 chunks
                    progress = (start_idx / num_current_patches) * 100
                    print(f"   Progress: {progress:.1f}% ({start_idx}/{num_current_patches})")
                
                try:
                    # Calcola similarità per questo chunk
                    chunk_similarities = torch.mm(current_chunk, goal_feat_norm.t())
                    
                    # Trova i migliori match per questo chunk
                    chunk_best_indices = torch.argmax(chunk_similarities, dim=1)
                    chunk_best_similarities = torch.max(chunk_similarities, dim=1)[0]
                    
                    # Memorizza i risultati
                    best_goal_indices[start_idx:end_idx] = chunk_best_indices
                    best_similarities_cg[start_idx:end_idx] = chunk_best_similarities
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"❌ CUDA OOM nel chunk {i}, provo con chunk size ridotto")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Prova con chunk size dimezzato
                    reduced_chunk_size = chunk_size // 2
                    if reduced_chunk_size < 10:
                        raise RuntimeError("Impossibile processare anche con chunk molto piccoli")
                    
                    # Riprocessa questo chunk con dimensione ridotta
                    for sub_start in range(start_idx, end_idx, reduced_chunk_size):
                        sub_end = min(sub_start + reduced_chunk_size, end_idx)
                        sub_chunk = current_feat_norm[sub_start:sub_end]
                        
                        sub_similarities = torch.mm(sub_chunk, goal_feat_norm.t())
                        sub_best_indices = torch.argmax(sub_similarities, dim=1)
                        sub_best_similarities = torch.max(sub_similarities, dim=1)[0]
                        
                        best_goal_indices[sub_start:sub_end] = sub_best_indices
                        best_similarities_cg[sub_start:sub_end] = sub_best_similarities
                        
                        del sub_similarities
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Libera memoria del chunk
                if 'chunk_similarities' in locals():
                    del chunk_similarities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Errore durante Current->Goal matching: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return None, None
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
        
        # 🔧 FIX: Usa dimensioni reali delle immagini invece di hardcoded
        load_h, load_w = self.load_size
        
        # Ottieni dimensioni reali dell'immagine goal
        if isinstance(goal_image, (str, Path)):
            pil_img = Image.open(goal_image)
            original_w, original_h = pil_img.size
        elif isinstance(goal_image, Image.Image):
            original_w, original_h = goal_image.size
        elif isinstance(goal_image, np.ndarray):
            # Per numpy arrays
            if len(goal_image.shape) == 3:
                original_h, original_w = goal_image.shape[:2]
            else:
                original_h, original_w = goal_image.shape
        else:
            # Fallback per altri tipi
            original_h, original_w = 480, 640  # Default fallback
        
        print(f"🔍 DEBUG: Original image size: {original_w}x{original_h}")
        print(f"🔍 DEBUG: Load size: {load_w}x{load_h}")
        print(f"🔍 DEBUG: Patches grid: {w_patches}x{h_patches}")
        print(f"🔍 DEBUG: Stride: {stride_w}x{stride_h}, Patch size: {patch_size}")
        
        scale_h = original_h / load_h
        scale_w = original_w / load_w
        print(f"🔍 DEBUG: Scale factors: w={scale_w:.3f}, h={scale_h:.3f}")
        
        goal_points = []
        current_points = []
        
        for i, (goal_idx, current_idx) in enumerate(best_matches):
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
            goal_x_scaled = goal_x * scale_w
            goal_y_scaled = goal_y * scale_h
            current_x_scaled = current_x * scale_w
            current_y_scaled = current_y * scale_h
            
            # Coordinate nel formato corretto [x, y] per matplotlib
            goal_points.append([goal_x_scaled, goal_y_scaled])
            current_points.append([current_x_scaled, current_y_scaled])
            
            # Debug per i primi 3 punti
            if i < 3:
                print(f"🔍 Point {i+1}: goal_patch({goal_patch_x},{goal_patch_y}) → pixel({goal_x_scaled:.1f},{goal_y_scaled:.1f})")
                print(f"              current_patch({current_patch_x},{current_patch_y}) → pixel({current_x_scaled:.1f},{current_y_scaled:.1f})")
        
        goal_points = np.array(goal_points)
        current_points = np.array(current_points)
        
        print(f"✅ ViT feature matching completato: {len(best_matches)} corrispondenze")
        avg_similarity = np.mean([match_data[i][1] for i in range(num_matches_to_use)])
        print(f"📊 Similarità media: {avg_similarity:.4f}")
        
        # Pulizia finale della memoria
        del goal_feat, current_feat, goal_feat_norm, current_feat_norm
        del best_current_indices, best_similarities_gc, best_goal_indices, best_similarities_cg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return goal_points, current_points
    
    def _smart_crop_object(self, pil_image: Image.Image, padding_ratio: float = 0.1) -> Image.Image:
        """Smart crop to focus on the main object by removing background/empty areas"""
        # Convert to numpy array for processing
        img_array = np.array(pil_image)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive threshold for better object detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: use edge detection if no contours found
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠️  No object detected, returning original image")
            return pil_image
        
        # Find the largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding around the object
        img_h, img_w = img_array.shape[:2]
        padding_w = int(w * padding_ratio)
        padding_h = int(h * padding_ratio)
        
        # Calculate crop coordinates with padding
        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(img_w, x + w + padding_w)
        y2 = min(img_h, y + h + padding_h)
        
        # Ensure minimum size for the crop
        min_size = 200  # Minimum crop size
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            # If object is too small, expand the crop area
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            half_size = max(min_size // 2, (x2 - x1) // 2, (y2 - y1) // 2)
            
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(img_w, center_x + half_size)
            y2 = min(img_h, center_y + half_size)
        
        # Crop the image
        cropped_img = img_array[y1:y2, x1:x2]
        
        # Convert back to PIL Image
        return Image.fromarray(cropped_img)

    # ...existing preprocess_image method...