"""
Utility Functions Module
Funzioni di utilità per visualizzazione e configurazione
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image
import yaml
from pathlib import Path
from util.data import Data

data = Data()
# Configura matplotlib per ambiente SSH/headless
if os.environ.get('DISPLAY') is None or os.environ.get('MPLBACKEND') == 'Agg':
    matplotlib.use('Agg')  # Backend non-interattivo per SSH
else:
    try:
        matplotlib.use('Qt5Agg')  # Backend interattivo per X11
    except ImportError:
        matplotlib.use('Agg')  # Fallback


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

    for i, ((x1, y1), (x2, y2), color) in enumerate(zip(points1, points2, colors)):
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
        os.makedirs(os.path.dirname(f"{save_path}"), exist_ok=True)
        data.last_result_path = f"{save_path}"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📸 Corrispondenze salvate: {save_path}")
    
    # Mostra solo se display disponibile
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig


def create_example_config():
    """Crea un file di configurazione di esempio"""
    config = {
        # GPU Configuration
        'device': None,  # Auto-detect, or specify: 'cuda:0', 'cuda:1', 'cpu', etc.

        # Camera parameters
        'u_max': 640,
        'v_max': 480,
        'f_x': 554.25,
        'f_y': 554.25,

        # Control parameters
        'lambda_': 0.5,
        'max_velocity': 1.0,
        'num_pairs': 20,  # More features for better accuracy

        'dino_input_size': 672,  # Higher resolution
        'model_type': 'dinov2_vits14',  # Larger model

        # Quality parameters
        'min_error': 3.0,
        'max_error': 150.0,
        'velocity_convergence_threshold': 0.05,
        'max_iterations': 2000,
        'min_iterations': 100,

        # Memory parameters (A6000 specific)
        'enable_memory_efficient': False,  # Disable sampling
        'similarity_threshold': 0.85,
        'bidirectional_matching': True,
        'feature_normalization': True,

        # Visualization
        'save_visualizations': True,
        'visualization_dpi': 300,
        'show_debug_info': True
    }
    
    config_path = "vitvs_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"File di configurazione creato: {config_path}")
    return config_path


def load_config(config_path: str) -> dict:
    """Carica configurazione da file YAML"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
