"""
ViT-VS Standalone Implementation - NON-ROS Version
Sistema standalone per Visual Servoing basato su Vision Transformer
Riceve goal image e current image, restituisce velocità di controllo IBVS
"""

import os
import warnings
import numpy as np
import torch
from typing import Optional, Union, Tuple
from pathlib import Path
from PIL import Image

# Import dei moduli separati
from modules.vit_extractor import ViTExtractor
from modules.utils import visualize_correspondences, load_config

warnings.filterwarnings("ignore")


class ViTVisualServoing:
    """Classe principale per Visual Servoing con ViT - Versione Standalone"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza il sistema ViT Visual Servoing standalone
        
        Args:
            config_path: Path al file di configurazione YAML (opzionale)
        """
        # Parametri di default (compatibili con sistema originale)
        self.default_params = {
            'u_max': 640,                     # Larghezza immagine
            'v_max': 480,                     # Altezza immagine  
            'f_x': 554.25,                    # Lunghezza focale X
            'f_y': 554.25,                    # Lunghezza focale Y
            'lambda_': 0.5,                   # Gain controllo IBVS
            'max_velocity': 1.0,              # Velocità massima
            'num_pairs': 20,                  # Numero feature da estrarre
            'dino_input_size': 518,           # Dimensione input DINOv2
            'model_type': 'dinov2_vits14',    # Tipo modello ViT
            'stride': None,                   # Auto-calculate compatible stride
            'device': None,                   # Auto-detect GPU/CPU
            'use_depth_default': True,        # Usa depth di default
            'default_depth': 1.0,             # Profondità di default (metri)
        }
        
        # Carica parametri da config se disponibile
        if config_path and Path(config_path).exists():
            try:
                config = load_config(config_path)
                for key, default_value in self.default_params.items():
                    setattr(self, key, config.get(key, default_value))
                print(f"✅ Configurazione caricata da {config_path}")
            except Exception as e:
                print(f"⚠️  Errore caricamento config: {e}, uso parametri default")
                for key, value in self.default_params.items():
                    setattr(self, key, value)
        else:
            print("📝 Usando parametri di default")
            for key, value in self.default_params.items():
                setattr(self, key, value)
        
        # Calcola parametri camera derivati
        self.c_x = self.u_max / 2  # Principal point X
        self.c_y = self.v_max / 2  # Principal point Y
        
        # Inizializza ViT Extractor
        try:
            self.vit_extractor = ViTExtractor(
                model_type=self.model_type, 
                stride=self.stride,
                device=self.device
            )
            print(f"✅ ViT Extractor inizializzato: {self.model_type} (stride: {self.stride or 'auto'})")
        except Exception as e:
            print(f"❌ Errore inizializzazione ViT: {e}")
            raise
        
        # Variabili di stato
        self.last_velocity = np.zeros(6)
        self.iteration_count = 0
    
    def detect_features(self, goal_image: Union[str, Image.Image], 
                       current_image: Union[str, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rileva feature corrispondenti tra goal image e current image
        
        Args:
            goal_image: Path o PIL Image della goal image
            current_image: Path o PIL Image della current image
            
        Returns:
            Tuple[points_goal, points_current]: Array dei punti corrispondenti
        """
        return self.vit_extractor.detect_vit_features(
            goal_image, current_image, 
            num_pairs=self.num_pairs,
            dino_input_size=self.dino_input_size
        )
    
    def transform_to_normalized_coordinates(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        Trasforma punti pixel in coordinate normalizzate della camera
        
        Args:
            pixel_points: Array di punti pixel [N, 2] formato [x, y]
            
        Returns:
            normalized_points: Array di punti normalizzati [N, 2] formato [x, y]
        """
        normalized_points = np.zeros_like(pixel_points, dtype=float)
        
        for i, (x, y) in enumerate(pixel_points):
            # Trasformazione da pixel a coordinate normalizzate
            norm_x = (x - self.c_x) / self.f_x
            norm_y = (y - self.c_y) / self.f_y
            normalized_points[i] = [norm_x, norm_y]
        
        return normalized_points
    
    def calculate_interaction_matrix(self, normalized_points: np.ndarray, 
                                   depths: np.ndarray) -> np.ndarray:
        """
        Calcola la matrice di interazione per IBVS
        
        Args:
            normalized_points: Punti in coordinate normalizzate [N, 2]
            depths: Profondità corrispondenti [N, 1] o [N]
            
        Returns:
            L: Matrice di interazione [2N, 6]
        """
        num_points = len(normalized_points)
        L = np.zeros([2 * num_points, 6], dtype=float)
        
        # Assicurati che depths sia un array 1D
        if depths.ndim > 1:
            depths = depths.flatten()
        
        for i, (x, y) in enumerate(normalized_points):
            z = depths[i] if i < len(depths) else self.default_depth
            
            # Assicurati che z non sia zero
            if z <= 0:
                z = self.default_depth
            
            # Calcola righe della matrice di interazione
            # Riga per coordinata x
            L[2*i, :] = [-1/z, 0, x/z, x*y, -(1 + x**2), y]
            # Riga per coordinata y  
            L[2*i + 1, :] = [0, -1/z, y/z, 1 + y**2, -x*y, -x]
        
        return L
    
    def compute_control_velocity(self, goal_points: np.ndarray, 
                               current_points: np.ndarray,
                               depths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcola la velocità di controllo IBVS
        
        Args:
            goal_points: Punti goal in pixel [N, 2]
            current_points: Punti correnti in pixel [N, 2] 
            depths: Profondità opzionali [N] (se None usa default)
            
        Returns:
            velocity: Vettore velocità [vx, vy, vz, ωx, ωy, ωz]
        """
        num_points = len(goal_points)
        
        # Usa profondità di default se non fornite
        if depths is None:
            depths = np.full(num_points, self.default_depth)
        elif len(depths) != num_points:
            # Adatta dimensioni se necessario
            depths = np.full(num_points, self.default_depth)
        
        # Trasforma in coordinate normalizzate
        goal_normalized = self.transform_to_normalized_coordinates(goal_points)
        current_normalized = self.transform_to_normalized_coordinates(current_points)
        
        # Calcola errore feature
        error = current_normalized - goal_normalized
        error_vector = error.reshape((-1, 1))  # [2N, 1]
        
        # Calcola matrice di interazione
        L = self.calculate_interaction_matrix(current_normalized, depths)
        
        # Calcola velocità con pseudoinversa
        try:
            L_pinv = np.linalg.pinv(L)
            velocity = -self.lambda_ * (L_pinv @ error_vector).flatten()
            
            # Applica limiti di velocità
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            
            # Salva per debug
            self.last_velocity = velocity.copy()
            
            return velocity
            
        except np.linalg.LinAlgError as e:
            print(f"⚠️  Errore calcolo velocità: {e}")
            return np.zeros(6)
    
    def process_image_pair(self, goal_image_path: Union[str, Path], 
                         current_image_path: Union[str, Path],
                         depths: Optional[np.ndarray] = None,
                         visualize: bool = True, 
                         save_path: Optional[str] = None) -> Optional[dict]:
        """
        Processa una coppia di immagini e calcola velocità di controllo
        
        Args:
            goal_image_path: Path alla goal image
            current_image_path: Path alla current image
            depths: Array profondità opzionale
            visualize: Se visualizzare le corrispondenze
            save_path: Path per salvare visualizzazione
            
        Returns:
            dict: Risultati con velocità, punti e informazioni aggiuntive
        """
        try:
            # Carica immagini 
            if isinstance(goal_image_path, (str, Path)):
                goal_image = Image.open(goal_image_path).convert('RGB')
            else:
                goal_image = goal_image_path
                
            if isinstance(current_image_path, (str, Path)):
                current_image = Image.open(current_image_path).convert('RGB')
            else:
                current_image = current_image_path
            
            print(f"🔍 Processando: {Path(str(goal_image_path)).name} -> {Path(str(current_image_path)).name}")
            print(f"📐 Metodo: ViT Visual Servoing (DINOv2)")
            
            # Rileva feature corrispondenti
            points_goal, points_current = self.detect_features(goal_image, current_image)
            
            if points_goal is None or points_current is None:
                print("❌ Fallimento nel rilevamento delle feature")
                return None
            
            num_features = len(points_goal)
            print(f"✅ Feature rilevate: {num_features} corrispondenze")
            
            # Calcola velocità di controllo IBVS
            velocity = self.compute_control_velocity(points_goal, points_current, depths)
            
            # Risultati dettagliati
            print(f"🎯 Velocità di controllo calcolata:")
            print(f"   Traslazione: vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}")
            print(f"   Rotazione:   ωx={velocity[3]:.4f}, ωy={velocity[4]:.4f}, ωz={velocity[5]:.4f}")
            
            # Calcola norma per valutazione
            velocity_norm = np.linalg.norm(velocity)
            print(f"📊 Norma velocità: {velocity_norm:.4f}")
            
            # Visualizza corrispondenze se richiesto
            if visualize:
                try:
                    fig = visualize_correspondences(
                        goal_image, current_image,
                        points_goal, points_current,
                        save_path
                    )
                    print(f"📸 Visualizzazione creata")
                    if save_path:
                        print(f"💾 Salvata in: {save_path}")
                except Exception as e:
                    print(f"⚠️  Errore visualizzazione: {e}")
            
            # Incrementa contatore
            self.iteration_count += 1
            
            return {
                'velocity': velocity,
                'goal_points': points_goal,
                'current_points': points_current, 
                'num_features': num_features,
                'velocity_norm': velocity_norm,
                'method': 'vit_standalone',
                'iteration': self.iteration_count
            }
            
        except Exception as e:
            print(f"❌ Errore elaborazione: {e}")
            return None
    
    def get_system_info(self) -> dict:
        """Restituisce informazioni sul sistema"""
        return {
            'camera_params': {
                'u_max': self.u_max,
                'v_max': self.v_max,
                'f_x': self.f_x,
                'f_y': self.f_y,
                'c_x': self.c_x,
                'c_y': self.c_y
            },
            'control_params': {
                'lambda': self.lambda_,
                'max_velocity': self.max_velocity,
                'num_pairs': self.num_pairs
            },
            'vit_params': {
                'model_type': self.model_type,
                'input_size': self.dino_input_size,
                'device': self.vit_extractor.device
            },
            'iteration_count': self.iteration_count
        }


if __name__ == "__main__":
    # Esempio di utilizzo standalone
    print("🎯 ViT Visual Servoing - Standalone System")
    print("=" * 50)
    
    try:
        # Inizializza sistema
        vit_vs = ViTVisualServoing()
        
        # Mostra informazioni sistema
        info = vit_vs.get_system_info()
        print("\n📋 Sistema inizializzato:")
        print(f"   Camera: {info['camera_params']['u_max']}x{info['camera_params']['v_max']}")
        print(f"   Focal: fx={info['camera_params']['f_x']}, fy={info['camera_params']['f_y']}")
        print(f"   Control gain: λ={info['control_params']['lambda']}")
        print(f"   Features: {info['control_params']['num_pairs']}")
        print(f"   ViT model: {info['vit_params']['model_type']}")
        print(f"   Device: {info['vit_params']['device']}")
        
        print("\n🔧 Utilizzo:")
        print("   result = vit_vs.process_image_pair(goal_path, current_path)")
        print("   velocity = result['velocity']  # [vx, vy, vz, ωx, ωy, ωz]")
        print("\n🎯 Sistema pronto per elaborazione immagini!")
        
    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
