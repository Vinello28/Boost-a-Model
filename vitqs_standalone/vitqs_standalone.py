"""
ViT-VS Standalone Implementation
Estratto dal progetto ROS per lavorare con immagini di dataset
"""

import os
import numpy as np
import cv2
from PIL import Image

# Configura matplotlib per ambiente SSH/headless
import matplotlib
if os.environ.get('DISPLAY') is None or os.environ.get('MPLBACKEND') == 'Agg':
    matplotlib.use('Agg')  # Backend non-interattivo per SSH
else:
    try:
        matplotlib.use('Qt5Agg')  # Backend interattivo per X11
    except ImportError:
        matplotlib.use('Agg')  # Fallback
        
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
import torch.nn.functional as F
import timm
from torchvision import transforms
import types
import math
from typing import Union, Tuple, List, Optional
from pathlib import Path
import yaml
import warnings

warnings.filterwarnings("ignore")


class ViTExtractor:
    """ViT feature extractor for DINOv2"""
    
    def __init__(self, model_type: str = 'dinov2_vits14', stride: int = 2, device: str = 'cuda'):
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        
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


def visualize_correspondences(image1, image2, points1, points2, save_path=None, show_plot=None):
    """Visualizza le corrispondenze tra due immagini"""
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    points1 = np.array(points1)
    points2 = np.array(points2)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(image1)
    ax2.imshow(image2)
    ax1.set_title("Goal Image")
    ax2.set_title("Current Image")

    ax1.axis('off')
    ax2.axis('off')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(points1)))

    for i, ((y1, x1), (y2, x2), color) in enumerate(zip(points1, points2, colors)):
        ax1.plot(x1, y1, 'o', color=color, markersize=8)
        ax1.text(x1 + 5, y1 + 5, str(i), color=color, fontsize=8)

        ax2.plot(x2, y2, 'o', color=color, markersize=8)
        ax2.text(x2 + 5, y2 + 5, str(i), color=color, fontsize=8)

        con = ConnectionPatch(
            xyA=(x1, y1), xyB=(x2, y2),
            coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color=color, alpha=0.5
        )
        fig.add_artist(con)

    plt.tight_layout()
    
    # Auto-detect se salvare o mostrare
    if show_plot is None:
        show_plot = os.environ.get('DISPLAY') is not None and matplotlib.get_backend() != 'Agg'
    
    # Salva sempre se richiesto o se in modalità headless
    if save_path or not show_plot:
        if save_path is None:
            save_path = 'results/correspondences.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📸 Corrispondenze salvate: {save_path}")
    
    # Mostra solo se display disponibile
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig


class ViTVisualServoing:
    """Classe principale per Visual Servoing con ViT"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Parametri di default
        self.default_params = {
            'u_max': 640,
            'v_max': 480,
            'f_x': 554.25,
            'f_y': 554.25,
            'lambda_': 0.5,
            'max_velocity': 1.0,
            'num_pairs': 10,
            'dino_input_size': 518,
            'model_type': 'dinov2_vits14',
            'min_error': 5.0,
            'max_error': 100.0,
            'velocity_convergence_threshold': 0.1,
            'max_iterations': 1000,
            'min_iterations': 50,
            'max_patches': 100000,
            'similarity_threshold': 0.8,
            'enable_memory_efficient': True
        }
        
        if config_path and Path(config_path).exists():
            self.load_parameters(config_path)
        else:
            print("Using default parameters")
            for key, value in self.default_params.items():
                setattr(self, key, value)
        
        # Parametri camera calcolati
        self.c_x = self.u_max / 2
        self.c_y = self.v_max / 2
        
        # Inizializza ViT extractor con modello configurabile
        model_type = getattr(self, 'model_type', 'dinov2_vits14')
        self.vit_extractor = ViTExtractor(model_type=model_type)
        
        # Variabili di controllo
        self.velocity_history = []
        self.iteration_count = 0
        
    def load_parameters(self, config_path: str):
        """Carica parametri da file YAML"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        for key, default_value in self.default_params.items():
            setattr(self, key, config.get(key, default_value))
    
    def detect_traditional_features(self, goal_image, current_image, method='sift'):
        """Rileva feature tradizionali (SIFT, ORB, AKAZE)"""
        if isinstance(goal_image, Image.Image):
            goal_image = np.array(goal_image)
        if isinstance(current_image, Image.Image):
            current_image = np.array(current_image)
        
        # Converti in grayscale
        goal_gray = cv2.cvtColor(goal_image, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        
        # Inizializza rilevatore
        if method.lower() == 'sift':
            detector = cv2.SIFT_create()
            norm_type = cv2.NORM_L2
        elif method.lower() == 'orb':
            detector = cv2.ORB_create(nfeatures=1000)
            norm_type = cv2.NORM_HAMMING
        elif method.lower() == 'akaze':
            detector = cv2.AKAZE_create()
            norm_type = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Metodo non supportato: {method}")
        
        # Rileva e calcola keypoints e descriptors
        kp1, des1 = detector.detectAndCompute(goal_gray, None)
        kp2, des2 = detector.detectAndCompute(current_gray, None)
        
        if des1 is None or des2 is None:
            print("Nessun descriptor trovato in una o entrambe le immagini")
            return None, None
        
        # Matcher
        bf = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 4:
            print(f"Matches insufficienti: {len(matches)} < 4")
            return None, None
        
        # Ordina per distanza
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Seleziona i migliori matches
        num_pairs_to_use = min(self.num_pairs, len(matches))
        matches = matches[:num_pairs_to_use]
        
        # Estrai punti
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        return points1, points2
    
    def detect_vit_features(self, goal_image, current_image, num_pairs=None):
        """Rileva feature usando ViT con matching reale basato su similarità coseno"""
        if num_pairs is None:
            num_pairs = self.num_pairs
            
        # Estrai feature ViT da entrambe le immagini usando token features
        goal_feats = self.vit_extractor.extract_features(
            goal_image, 
            load_size=self.dino_input_size, 
            facet='token'  # Usa token features invece di key/query/value
        )
        current_feats = self.vit_extractor.extract_features(
            current_image, 
            load_size=self.dino_input_size,
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
        
        # RTX A6000 (48GB VRAM) - No memory limitations needed
        print(f"� Processing {goal_feat.shape[0]} patches with RTX A6000")
        
        # Normalizza le feature per similarità coseno
        goal_feat_norm = F.normalize(goal_feat, dim=-1)
        current_feat_norm = F.normalize(current_feat, dim=-1)
        
        # Calcola matrice di similarità coseno
        similarity_matrix = torch.mm(goal_feat_norm, current_feat_norm.t())  # [N_goal, N_current]
        
        # Trova le corrispondenze migliori usando matching bidirezionale
        # Goal -> Current
        best_current_indices = torch.argmax(similarity_matrix, dim=1)
        best_similarities_gc = torch.max(similarity_matrix, dim=1)[0]
        
        # Current -> Goal 
        best_goal_indices = torch.argmax(similarity_matrix, dim=0)
        best_similarities_cg = torch.max(similarity_matrix, dim=0)[0]
        
        # Trova matches consistenti (corrispondenze bidirezionali)
        consistent_matches = []
        similarities = []
        
        for i in range(len(best_current_indices)):
            j = best_current_indices[i].item()
            if best_goal_indices[j].item() == i:  # Match bidirezionale
                # Senza campionamento, usa indici diretti
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
        h_patches, w_patches = self.vit_extractor.num_patches
        stride_h, stride_w = self.vit_extractor.stride[0], self.vit_extractor.stride[1]
        patch_size = self.vit_extractor.p
        
        # Calcola il fattore di scala dalle dimensioni originali
        load_h, load_w = self.vit_extractor.load_size
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
    
    def transform_to_real_world(self, s_uv, s_uv_star):
        """Trasforma punti pixel in coordinate mondo reale"""
        s_xy = []
        s_star_xy = []
        
        for uv, uv_star in zip(s_uv, s_uv_star):
            x = (uv[0] - self.c_x) / self.f_x
            y = (uv[1] - self.c_y) / self.f_y
            s_xy.append([x, y])
            
            x_star = (uv_star[0] - self.c_x) / self.f_x
            y_star = (uv_star[1] - self.c_y) / self.f_y
            s_star_xy.append([x_star, y_star])
        
        return np.array(s_xy), np.array(s_star_xy)
    
    def calculate_interaction_matrix(self, s_xy, depths):
        """Calcola la matrice di interazione per i punti feature"""
        L = np.zeros([2 * len(s_xy), 6], dtype=float)
        
        for count in range(len(s_xy)):
            x, y = s_xy[count, 0], s_xy[count, 1]
            z = depths[count] if isinstance(depths, (list, np.ndarray)) else depths
            
            L[2 * count, :] = [-1/z, 0, x/z, x*y, -(1 + x**2), y]
            L[2 * count + 1, :] = [0, -1/z, y/z, 1 + y**2, -x*y, -x]
        
        return L
    
    def compute_velocity(self, goal_image, current_image, depths=None, method='sift'):
        """Calcola velocità di controllo IBVS"""
        # Rileva feature
        if method.lower() in ['sift', 'orb', 'akaze']:
            points_goal, points_current = self.detect_traditional_features(
                goal_image, current_image, method
            )
        elif method.lower() == 'vit':
            points_goal, points_current = self.detect_vit_features(
                goal_image, current_image
            )
        else:
            raise ValueError(f"Metodo non supportato: {method}")
        
        if points_goal is None or points_current is None:
            return None, None, None
        
        # Trasforma in coordinate mondo reale
        s_xy, s_star_xy = self.transform_to_real_world(points_current, points_goal)
        
        # Calcola errore
        e = s_xy - s_star_xy
        e = e.reshape((len(s_xy) * 2, 1))
        
        # Profondità (default o fornita)
        if depths is None:
            depths = np.ones(len(s_xy)) * 1.0  # 1 metro di default
        
        # Calcola matrice di interazione
        L = self.calculate_interaction_matrix(s_xy, depths)
        
        # Calcola velocità
        v_c = -self.lambda_ * np.linalg.pinv(L.astype('float')) @ e
        
        # Applica limiti
        v_c = np.clip(v_c, -self.max_velocity, self.max_velocity)
        
        return v_c.flatten(), points_goal, points_current
    
    def process_image_pair(self, goal_image_path, current_image_path, 
                          depths=None, method='sift', visualize=True, save_path=None):
        """Processa una coppia di immagini e calcola la velocità di controllo"""
        
        # Carica immagini
        goal_image = Image.open(goal_image_path).convert('RGB')
        current_image = Image.open(current_image_path).convert('RGB')
        
        print(f"Processando: {Path(goal_image_path).name} -> {Path(current_image_path).name}")
        print(f"Metodo: {method.upper()}")
        
        # Calcola velocità
        velocity, points_goal, points_current = self.compute_velocity(
            goal_image, current_image, depths, method
        )
        
        if velocity is None:
            print("❌ Fallimento nel rilevamento delle feature")
            return None
        
        # Risultati
        print(f"✅ Feature rilevate: {len(points_goal)} coppie")
        print(f"🎯 Velocità calcolata:")
        print(f"   Traslazione: vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}")
        print(f"   Rotazione:   ωx={velocity[3]:.4f}, ωy={velocity[4]:.4f}, ωz={velocity[5]:.4f}")
        
        # Visualizza corrispondenze
        if visualize and points_goal is not None:
            visualize_correspondences(
                goal_image, current_image, 
                points_goal, points_current,
                save_path
            )
        
        return {
            'velocity': velocity,
            'goal_points': points_goal,
            'current_points': points_current,
            'num_features': len(points_goal),
            'method': method
        }


def create_example_config():
    """Crea un file di configurazione di esempio"""
    config = {
        'u_max': 640,
        'v_max': 480,
        'f_x': 554.25,
        'f_y': 554.25,
        'lambda_': 0.5,
        'max_velocity': 1.0,
        'num_pairs': 10,
        'dino_input_size': 518,
        'min_error': 5.0,
        'max_error': 100.0,
        'velocity_convergence_threshold': 0.1,
        'max_iterations': 1000,
        'min_iterations': 50
    }
    
    config_path = "vitqs_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"File di configurazione creato: {config_path}")
    return config_path


if __name__ == "__main__":
    # Esempio di utilizzo
    print("🤖 ViT-VS Standalone System")
    print("=" * 50)
    
    # Crea configurazione di esempio
    config_path = create_example_config()
    
    # Inizializza sistema
    vitqs = ViTVisualServoing(config_path)
    
    print("\n📋 Sistema inizializzato con parametri:")
    print(f"   Risoluzione immagine: {vitqs.u_max}x{vitqs.v_max}")
    print(f"   Lunghezza focale: fx={vitqs.f_x}, fy={vitqs.f_y}")
    print(f"   Gain controllo: λ={vitqs.lambda_}")
    print(f"   Numero feature: {vitqs.num_pairs}")
    
    print("\n🔧 Per utilizzare il sistema:")
    print("1. vitqs.process_image_pair(goal_path, current_path, method='sift')")
    print("2. Metodi disponibili: 'sift', 'orb', 'akaze', 'vit'")
    print("3. Il sistema ritorna velocità di controllo [vx, vy, vz, ωx, ωy, ωz]")
