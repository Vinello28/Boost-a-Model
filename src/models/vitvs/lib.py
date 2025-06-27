"""
ViT-VS Library
"""
import warnings
import logging
logger = logging.getLogger(__name__)

import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path
import numpy.typing as npt
from models.vitvs.modules.vit_extractor import ViTExtractor
from models.vitvs.modules.ibvs_controller import IBVSController
from models.vitvs.modules.utils import visualize_correspondences, create_example_config, load_config

warnings.filterwarnings("ignore")

class ProcessFrameResult:
    def __init__(
        self,
        velocity: Optional[npt.NDArray[np.float64]] = None,
        velocity_norm: Optional[float] = None,
        points_goal: Optional[npt.NDArray[np.float64]] = None,
        points_current: Optional[npt.NDArray[np.float64]] = None,
        num_features: Optional[int] = None,
    ) -> None:
        self.velocity = velocity
        self.velocity_norm = velocity_norm
        self.points_goal = points_goal
        self.points_current = points_current
        self.num_features = num_features

    def __repr__(self) -> str:
        return f"ProcessFrameResult(velocity={self.velocity}, num_features={self.num_features})"



class VitVsLib:
    def __init__(self, config_path: Optional[str] = None, gui: bool = True, device="cuda:0"):
        # default values for parameters
        self.gui = gui
        self.u_max= 640
        self.v_max= 480
        self.f_x= 554.25
        self.f_y= 554.25
        self.lambda_= 0.5
        self.max_velocity= 1.0
        self.num_pairs= 10
        self.dino_input_size= None # Dinamically set
        self.model_type= "dinov2_vits14"
        self.device= device
        self.min_error= 3.0
        self.max_error= 150.0
        self.velocity_convergence_threshold= 0.05
        self.max_iterations= 2000
        self.min_iterations= 100
        self.max_patches= 100000
        self.similarity_threshold= 0.85
        self.enable_memory_efficient= True
        self.bidirectional_matching = True
        self.feature_normalization = True

        # Load parameters from config if available
        if config_path and Path(config_path).exists():
            logging.info(f"Loading configuration from {config_path}")
            config = load_config(config_path)
            self.u_max = config.get("u_max", self.u_max)
            self.v_max = config.get("v_max", self.v_max)
            self.f_x = config.get("f_x", self.f_x)
            self.f_y = config.get("f_y", self.f_y)
            self.lambda_ = config.get("lambda_", self.lambda_)
            self.max_velocity = config.get("max_velocity", self.max_velocity)
            self.num_pairs = config.get("num_pairs", self.num_pairs)
            self.dino_input_size = config.get("dino_input_size", self.dino_input_size)
            self.model_type = config.get("model_type", self.model_type)
            self.device = config.get("device", self.device)
            self.min_error = config.get("min_error", self.min_error)
            self.max_error = config.get("max_error", self.max_error)
            self.velocity_convergence_threshold = config.get("velocity_convergence_threshold", self.velocity_convergence_threshold)
            self.max_iterations = config.get("max_iterations", self.max_iterations)
            self.min_iterations = config.get("min_iterations", self.min_iterations)
            self.max_patches = config.get("max_patches", self.max_patches)
            self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
            self.enable_memory_efficient = config.get("enable_memory_efficient", self.enable_memory_efficient)
            logging.info("Configuration loaded")
        else:
            logging.warning("No configuration found, using default parameters")

        if self.device == "cpu":
            logging.warning("Using CPU for processing, this may be slow. Consider using a GPU.")
        elif not self.device.startswith("cuda"):
            logging.info(f"Using GPU: {self.device}")  

        self.vit_extractor = ViTExtractor(model_type=self.model_type, device=self.device)

        self.ibvs_controller = IBVSController(
            u_max=self.u_max,
            v_max=self.v_max, 
            f_x=self.f_x,
            f_y=self.f_y,
            lambda_=self.lambda_,
            max_velocity=self.max_velocity
        )
        
        # Control variables
        self.velocity_history = []
        self.iteration_count = 0
         
    #TODO: set typing for goal and current frame
    def detect_features(self, goal_frame, current_frame):
        """Detects features using ViT"""

        return self.vit_extractor.detect_vit_features(
            goal_frame, current_frame, 
            num_pairs=self.num_pairs,
            dino_input_size=self.dino_input_size
        )
    
    #TODO: set typing for goal and current frame
    def compute_velocity(self, goal_frame, current_frame, depths=None):
        """Calculates IBVS control velocity using ViT"""

        # Get features
        points_goal, points_current = self.detect_features(
            goal_frame, current_frame
        )
        
        if points_goal is None or points_current is None:
            return None, None, None
        
        # Calculate velocity using the IBVS controller
        # Not strongly needed, but it serves as a way to visualize
        # how it would handle a servo
        velocity = self.ibvs_controller.compute_velocity(
            points_goal, points_current, depths
        )
        
        return velocity, points_goal, points_current
    
    def process_frame_pair(self, goal_image_path: Union[str, Path], 
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

            # IBVS control velocity
            velocity = self.ibvs_controller.compute_velocity(
                points_goal, points_current, depths
            )

            # IBVS control velocity normalization
            velocity_norm = np.linalg.norm(velocity)

            # Render correspondence matching
            if self.data.state.is_render_enabled:
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

            result = ProcessFrameResult(
                velocity=velocity,
                velocity_norm=velocity_norm,
                points_goal=points_goal,
                points_current=points_current,
                num_features=num_features,
            )

        except AssertionError as e:
            logging.error(f"Assertion error: {e}", exc_info=True)
            logging.info("exiting for safe debuigging")
            exit(os.EX_IOERR)
        except Exception as e:
            print(f"❌ Errore elaborazione: {e}")
            return None

    def depr_process_frame_pair(self,
            goal_frame,
            current_frame, 
            depths=None,
            save_path=None
        ):
        """Processes a pair of frames and computes control velocity """

        # Calculate velocity
        velocity, points_goal, points_current = self.compute_velocity(
            goal_frame, current_frame, depths
        )
        
        if velocity is None:
            logging.error(f"Failed to compute velocity for frames") 
            return None
        
        # Visualize correspondences if GUI is enabled
        if points_goal is not None:
            lpg = len(points_goal) 
            if self.gui:
                visualize_correspondences(
                    goal_frame, current_frame, 
                    points_goal, points_current,
                    save_path
                )
        else:
            lpg = 0
            logging.warning(f"No features detected in frames")

        return ProcessFrameResult(
            velocity=velocity,
            points_goal=points_goal,
            points_current=points_current,
            num_features=lpg
        )

        # return {
        #     'velocity': velocity,
        #     'goal_points': points_goal,
        #     'current_points': points_current,
        #     'num_features': lpg
        # }