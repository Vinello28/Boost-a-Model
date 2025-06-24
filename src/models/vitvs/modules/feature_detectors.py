"""
Traditional Feature Detectors Module
Gestisce i detector tradizionali (SIFT, ORB, AKAZE)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple


class TraditionalFeatureDetector:
    """Detector per feature tradizionali"""
    
    def __init__(self, num_pairs: int = 10):
        self.num_pairs = num_pairs
    
    def detect_features(self, goal_image, current_image, method='sift') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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