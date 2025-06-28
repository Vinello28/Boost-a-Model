"""
ViT-VS Library
"""

import os
import time
import warnings
import logging

from PIL import Image
import numpy as np

from util.data import Data
from typing import Optional
from pathlib import Path
import numpy.typing as npt
from models.vitvs.modules.vit_extractor import ViTExtractor
from models.vitvs.modules.ibvs_controller import IBVSController
from models.vitvs.modules.utils import (
    render_correspondence,
    create_example_config,
    load_config,
)
from util.metrics import MetricPoint, Metrics

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class ProcessFrameResult:
    def __init__(
        self,
        velocity: Optional[npt.NDArray[np.float64]] = None,
        velocity_norm: Optional[float] = None,
        points_goal: Optional[npt.NDArray[np.float64]] = None,
        points_current: Optional[npt.NDArray[np.float64]] = None,
        num_features: Optional[int] = None,
        execution_time: Optional[float] = None,
    ) -> None:
        self.velocity = velocity
        self.velocity_norm = velocity_norm
        self.points_goal = points_goal
        self.points_current = points_current
        self.num_features = num_features
        self.execution_time = execution_time

    def __repr__(self) -> str:
        return f"ProcessFrameResult(velocity={self.velocity}, num_features={self.num_features})"


class VitVsConfig:
    def __init__(self, gui, device) -> None:
        self.gui = gui
        self.u_max = 640
        self.v_max = 480
        self.f_x = 554.25
        self.f_y = 554.25
        self.lambda_ = 0.5
        self.max_velocity = 1.0
        self.num_pairs = 10
        self.dino_input_size = 518  # Dinamically set
        self.model_type = "dinov2_vits14"
        self.device = device
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

    def get_config_dict(self) -> dict:
        """Return controller configuration as a dictionary."""
        return {
            "gui": self.gui,
            "u_max": self.u_max,
            "v_max": self.v_max,
            "f_x": self.f_x,
            "f_y": self.f_y,
            "lambda": self.lambda_,
            "max_velocity": self.max_velocity,
            "num_pairs": self.num_pairs,
            "dino_input_size": self.dino_input_size,
            "model_type": self.model_type,
            "device": self.device,
            "min_error": self.min_error,
            "max_error": self.max_error,
            "velocity_convergence_threshold": self.velocity_convergence_threshold,
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
            "max_patches": self.max_patches,
            "similarity_threshold": self.similarity_threshold,
            "enable_memory_efficient": self.enable_memory_efficient,
            "bidirectional_matching": self.bidirectional_matching,
            "feature_normalization": self.feature_normalization,
        }


class VitVsLib:
    def __init__(
        self,
        config_path: Optional[str] = None,
        gui: bool = True,
        device: Optional[str] = "cuda:0",
        enable_metrics: Optional[bool] = None,
        metrics_save_path: Optional[str] = None,
    ):
        self.data = Data()

        # default values for parameters
        self.config = VitVsConfig(gui, device)

        if enable_metrics:
            self._is_metrics_enabled = enable_metrics
        else:
            self._is_metrics_enabled = self.data.state.is_metrics_enabled

        self._metrics_save_path = (
            metrics_save_path or f"metrics-for-{self.data.str_time_start}"
        )

        self.metrics = None

        # Load parameters from config if available
        if config_path and Path(config_path).exists():
            logging.info(f"Loading configuration from {config_path}")

            loaded_conf = load_config(config_path)
            self.config.u_max = loaded_conf.get("u_max", self.config.u_max)
            self.config.v_max = loaded_conf.get("v_max", self.config.v_max)
            self.config.f_x = loaded_conf.get("f_x", self.config.f_x)
            self.config.f_y = loaded_conf.get("f_y", self.config.f_y)
            self.config.lambda_ = loaded_conf.get("lambda_", self.config.lambda_)
            self.config.max_velocity = loaded_conf.get(
                "max_velocity", self.config.max_velocity
            )
            self.config.num_pairs = loaded_conf.get("num_pairs", self.config.num_pairs)
            self.config.dino_input_size = loaded_conf.get(
                "dino_input_size", self.config.dino_input_size
            )
            self.config.model_type = loaded_conf.get(
                "model_type", self.config.model_type
            )
            self.config.device = loaded_conf.get("device", self.config.device)
            self.config.min_error = loaded_conf.get("min_error", self.config.min_error)
            self.config.max_error = loaded_conf.get("max_error", self.config.max_error)
            self.config.velocity_convergence_threshold = loaded_conf.get(
                "velocity_convergence_threshold",
                self.config.velocity_convergence_threshold,
            )
            self.config.max_iterations = loaded_conf.get(
                "max_iterations", self.config.max_iterations
            )
            self.config.min_iterations = loaded_conf.get(
                "min_iterations", self.config.min_iterations
            )
            self.config.max_patches = loaded_conf.get(
                "max_patches", self.config.max_patches
            )
            self.config.similarity_threshold = loaded_conf.get(
                "similarity_threshold", self.config.similarity_threshold
            )
            self.config.enable_memory_efficient = loaded_conf.get(
                "enable_memory_efficient", self.config.enable_memory_efficient
            )
            logging.info("Configuration loaded")
        else:
            if self._is_metrics_enabled:
                self.metrics = Metrics(self.data.model_name, None)
            logging.warning("No configuration found, using default parameters")

        if self.config.device == "cpu":
            logging.warning(
                "Using CPU for processing, this may be slow. Consider using a GPU."
            )
        elif not str(self.config.device).startswith("cuda"):
            logging.info(f"Using GPU: {self.config.device}")

        if self._is_metrics_enabled:
            self.metrics = Metrics(self.data.model_name, self.config)

        self.vit_extractor = ViTExtractor(
            model_type=self.config.model_type,
            device=self.data.device or "cpu",
            stride=7,
        )

        self.ibvs_controller = IBVSController(
            u_max=self.config.u_max,
            v_max=self.config.v_max,
            f_x=self.config.f_x,
            f_y=self.config.f_y,
            lambda_=self.config.lambda_,
            max_velocity=self.config.max_velocity,
        )

        # Control variables
        self.velocity_history = []
        self.iteration_count = 0

    # TODO: set typing for goal and current frame
    def detect_features(self, gf: Image.Image, inf: Image.Image):
        """Detects features using ViT"""

        return

    # TODO: set typing for goal and current frame
    def compute_velocity(
        self,
        points_goal,
        points_current,
        depths: Optional[npt.NDArray] = None,
    ):
        """Calculates IBVS control velocity using ViT - get points using detect_features"""

        # Calculate velocity using the IBVS controller
        velocity = self.ibvs_controller.compute_velocity(
            points_goal, points_current, depths
        )

        return velocity, points_goal, points_current

    def get_metrics(self) -> Optional[Metrics]:
        """Returns the metrics object if metrics are enabled."""
        if self._is_metrics_enabled and isinstance(self.metrics, Metrics):
            return self.metrics
        return None

    def save_metrics(self, path: Optional[str] = None) -> None:
        """
        Saves the metrics to a file.
        """
        if self._is_metrics_enabled and isinstance(self.metrics, Metrics):
            if path is None:
                path = self._metrics_save_path
            self.metrics.save(path)
            logging.info(f"Metrics saved to {path}")
        else:
            logging.warning("Metrics are not enabled, nothing to save.")


    def process_frame_pair(
        self,
        gf: Image.Image,
        inf: Image.Image,
        depths: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Optional[ProcessFrameResult]:
        """
        Processes a pair of images and computes control velocities.

        Args:
            gf: Path to the goal image
            current_image_path: Path to the current image
            depths: Optional depth array
            visualize: Whether to display the matches
            save_path: Path to save the visualization

        Returns:
            ProcessFrameResult()
        """

        start = time.time()
        result = None
        try:
            # Rileva feature corrispondenti
            its = time.time()

            points_goal, points_current, metrics = (
                self.vit_extractor.detect_vit_features(
                    gf,
                    inf,
                    num_pairs=self.config.num_pairs,
                    dino_input_size=int(self.config.dino_input_size or 518),
                    enable_metrics=True,
                )
            )

            itf = time.time()
            if metrics:
                metrics.inference_time_ms = itf - its  # inference_time_ms

            if points_goal is None or points_current is None:
                logging.error("❌ Feature detection failed")
                logging.debug(f"     progress was {self.data.progress}")
                logging.debug(f"     gf position was {self.data.gf_position}")
                logging.debug(f"     inf position was {self.data.gf_position}")

                return None

            num_features = len(points_goal)

            if metrics:
                metrics.features_num = num_features

            # IBVS control velocity
            velocity = self.ibvs_controller.compute_velocity(
                points_goal, points_current, depths
            )

            if metrics:
                metrics.velocity_vector = velocity

            # IBVS control velocity normalization
            velocity_norm = np.linalg.norm(velocity)

            if metrics:
                metrics.velocity_norm = velocity_norm

            # Render correspondence matching
            if self.data.state.is_render_enabled:
                try:
                    render_correspondence(
                        gf,
                        inf,
                        points_goal,
                        points_current,
                        save_path,
                    )
                except Exception:
                    logging.error("Could not display frames", exc_info=True)
                    logging.debug(f"     progress was {self.data.progress}")
                    logging.debug(f"     gf position was {self.data.gf_position}")
                    logging.debug(f"     inf position was {self.data.gf_position}")

            # Incrementa contatore
            self.iteration_count += 1

            if (
                self._is_metrics_enabled
                and isinstance(self.metrics, Metrics)
                and metrics
            ):
                self.metrics.add(metrics)
                logging.info(f"Metrics added: {metrics}")

            finish = time.time()
            tp = finish - start
            self.data.add_time_point(tp)

            result = ProcessFrameResult(
                velocity=velocity,
                velocity_norm=velocity_norm,
                points_goal=points_goal,
                points_current=points_current,
                num_features=num_features,
                execution_time=tp,
            )

        except AssertionError as e:
            logging.error(f"Assertion error: {e}", exc_info=True)
            logging.info("exiting for safe debuigging")
            exit(os.EX_IOERR)
        except Exception as e:
            logging.error(f"Unexpected error occured: {e}", exc_info=True)
        finally:
            logging.info(
                f"frame {self.data.progress} (gf {self.data.gf_position}) (inf {self.data.inf_position}) took {tp} ms to process"
            )

        return result, metrics
