"""
ViT-VS Standalone Implementation - Modular Version
Classe principale che integra ViT extractor, feature detectors e IBVS controller
"""

import os
import warnings

from typing import Optional
from pathlib import Path
from PIL import Image

# Import dei moduli separati
from modules.vit_extractor import ViTExtractor
from modules.ibvs_controller import IBVSController
from modules.utils import visualize_correspondences, create_example_config, load_config

warnings.filterwarnings("ignore")


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
            'num_pairs': 20,
            'dino_input_size': 672,
            'model_type': 'dinov2_vits14',
            'device': None,  # Auto-detect or specify 'cuda:0', 'cuda:1', etc.
            'min_error': 5.0,
            'max_error': 100.0,
            'velocity_convergence_threshold': 0.1,
            'max_iterations': 1000,
            'min_iterations': 50,
            'max_patches': 100000,
            'similarity_threshold': 0.8,
            'enable_memory_efficient': True
        }
        
        # Carica parametri
        if config_path and Path(config_path).exists():
            config = load_config(config_path)
            for key, default_value in self.default_params.items():
                setattr(self, key, config.get(key, default_value))
        else:
            print("Using default parameters")
            for key, value in self.default_params.items():
                setattr(self, key, value)
        
        # Inizializza moduli
        self.vit_extractor = ViTExtractor(model_type=self.model_type, device=self.device)
        self.ibvs_controller = IBVSController(
            u_max=self.u_max,
            v_max=self.v_max, 
            f_x=self.f_x,
            f_y=self.f_y,
            lambda_=self.lambda_,
            max_velocity=self.max_velocity
        )
        
        # Variabili di controllo
        self.velocity_history = []
        self.iteration_count = 0
    
    def detect_features(self, goal_image, current_image, method='vit'):
        """Rileva feature usando ViT"""
        return self.vit_extractor.detect_vit_features(
            goal_image, current_image, 
            num_pairs=self.num_pairs,
            dino_input_size=self.dino_input_size
        )
    
    def compute_velocity(self, goal_image, current_image, depths=None, method='vit'):
        """Calcola velocità di controllo IBVS usando ViT"""
        # Rileva feature
        points_goal, points_current = self.detect_features(
            goal_image, current_image, method
        )
        
        if points_goal is None or points_current is None:
            return None, None, None
        
        # Calcola velocità usando IBVS controller
        velocity = self.ibvs_controller.compute_velocity(
            points_goal, points_current, depths
        )
        
        return velocity, points_goal, points_current
    
    def process_image_pair(self, goal_image_path, current_image_path, 
                          depths=None, method='vit', visualize=True, save_path=None):
        """Processa una coppia di immagini e calcola la velocità di controllo usando ViT"""
        
        # Carica immagini
        goal_image = Image.open(goal_image_path).convert('RGB')
        current_image = Image.open(current_image_path).convert('RGB')
        
        print(f"Processando: {Path(goal_image_path).name} -> {Path(current_image_path).name}")
        print(f"Metodo: ViT (Vision Transformer)")
        
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
            'method': 'vit'
        }


if __name__ == "__main__":
    # Esempio di utilizzo
    print("M O S  -  V S")
    print ("MOdular Sistem for Visual Servoing")
    print("=" * 50)
    
    # Crea configurazione di esempio
    config_path = "vitvs_config.yaml"
    
    # Inizializza sistema
    vitqs = ViTVisualServoing(config_path)
    
    print("\n📋 Sistema inizializzato con parametri:")
    print(f"   Risoluzione immagine: {vitqs.u_max}x{vitqs.v_max}")
    print(f"   Lunghezza focale: fx={vitqs.f_x}, fy={vitqs.f_y}")
    print(f"   Gain controllo: λ={vitqs.lambda_}")
    print(f"   Numero feature: {vitqs.num_pairs}")
    
    print("\n🔧 Per utilizzare il sistema:")
    print("1. vitqs.process_image_pair(goal_path, current_path)")
    print("2. Sistema basato esclusivamente su ViT (Vision Transformer)")
    print("3. Il sistema ritorna velocità di controllo [vx, vy, vz, ωx, ωy, ωz]")
    
    print("\n📁 Struttura modulare:")
    print("   - vit_extractor.py: Gestione ViT e feature extraction")
    print("   - ibvs_controller.py: Logica IBVS e controllo")
    print("   - utils.py: Funzioni di utilità e visualizzazione")
    print("   - vitqs_standalone.py: Classe principale ViT-VS")
