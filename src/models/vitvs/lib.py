"""
ViT-VS Library
"""

import warnings
import logging

from PIL import Image
import numpy as np

from util.data import Data
from typing import Optional
from pathlib import Path
from models.vitvs.modules.vit_extractor import ViTExtractor
from models.vitvs.modules.ibvs_controller import IBVSController
from models.vitvs.modules.utils import (
    visualize_correspondences,
    create_example_config,
    load_config,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class VitVsLib:
    def __init__(
        self,
        data: Data,
    ):
        self.data = data
        config_path = data.config_path

        # default values for parameters
        self.gui = data.state.is_gui_enabled
        self.u_max = 640
        self.v_max = 480
        self.f_x = 554.25
        self.f_y = 554.25
        self.lambda_ = 0.5
        self.max_velocity = 1.0
        self.num_pairs = 10
        self.dino_input_size = None  # Dinamically set
        self.model_type = "dinov2_vits14"
        self.device = data.device
        self.min_error = 3.0
        self.max_error = 150.0
        self.velocity_convergence_threshold = 0.05
        self.max_iterations = 2000
        self.min_iterations = 100
        self.max_patches = 100000
        self.similarity_threshold = 0.85
        self.enable_memory_efficient = True
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
            self.dino_input_size = config.get(
                "dino_input_size", self.dino_input_size
            )
            self.model_type = config.get("model_type", self.model_type)
            self.device = config.get("device", self.device)
            self.min_error = config.get("min_error", self.min_error)
            self.max_error = config.get("max_error", self.max_error)
            self.velocity_convergence_threshold = config.get(
                "velocity_convergence_threshold",
                self.velocity_convergence_threshold,
            )
            self.max_iterations = config.get("max_iterations", self.max_iterations)
            self.min_iterations = config.get("min_iterations", self.min_iterations)
            self.max_patches = config.get("max_patches", self.max_patches)
            self.similarity_threshold = config.get(
                "similarity_threshold", self.similarity_threshold
            )
            self.enable_memory_efficient = config.get(
                "enable_memory_efficient", self.enable_memory_efficient
            )
            logging.info("Configuration loaded")
        else:
            logging.warning("No configuration found, using default parameters")

        if self.device == "cpu":
            logging.warning(
                "Using CPU for processing, this may be slow. Consider using a GPU."
            )
        elif not str(self.device).startswith("cuda"):
            logging.info(f"Using GPU: {self.device}")

        self.vit_extractor = ViTExtractor(
            model_type=self.model_type, device=self.data.device or "cpu"
        )

        self.ibvs_controller = IBVSController(
            u_max=self.u_max,
            v_max=self.v_max,
            f_x=self.f_x,
            f_y=self.f_y,
            lambda_=self.lambda_,
            max_velocity=self.max_velocity,
        )

        # Control variables
        self.velocity_history = []
        self.iteration_count = 0

    def detect_features(self, goal_frame, current_frame):
        """Detects features using ViT"""

        return self.vit_extractor.detect_vit_features(
            goal_frame,
            current_frame,
            num_pairs=self.num_pairs,
            dino_input_size=int(self.dino_input_size or 518),
        )

    def compute_velocity(self, goal_frame, current_frame, depths=None):
        """Calculates IBVS control velocity using ViT"""

        points_goal, points_current = self.detect_features(goal_frame, current_frame)

        if points_goal is None or points_current is None:
            return None, None, None

        velocity = self.ibvs_controller.compute_velocity(
            points_goal, points_current, depths
        )

        return velocity, points_goal, points_current

    def process_frame_pair(
        self,
        gf: Image.Image,
        inf: Image.Image,
        depths: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Processes a pair of images and computes control velocities.
        """
        try:
            velocity, points_goal, points_current = self.compute_velocity(
                gf, inf, depths
            )

            if velocity is None:
                logging.error("❌ Feature detection failed")
                return None

            num_features = len(points_goal)
            logging.info(f"✅ Feature detected: {num_features} correspondences")

            if self.data.state.is_gui_enabled:
                try:
                    visualize_correspondences(
                        gf,
                        inf,
                        points_goal,
                        points_current,
                        save_path,
                    )
                except Exception as e:
                    logging.error("Could not display frames", exc_info=True)

            self.iteration_count += 1

            return {
                "velocity": velocity,
                "points_goal": points_goal,
                "points_current": points_current,
                "num_features": num_features,
            }

        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}", exc_info=True)
            return None

