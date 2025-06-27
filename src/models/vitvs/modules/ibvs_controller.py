"""
IBVS Controller Module
Gestisce la logica di controllo Image-Based Visual Servoing
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Union, List


class IBVSController:
    """Controller IBVS per calcolo velocità"""
    
    def __init__(self, 
                 u_max: int = 640, 
                 v_max: int = 480,
                 f_x: float = 554.25, 
                 f_y: float = 554.25,
                 lambda_: float = 0.5,
                 max_velocity: float = 1.0):
        
        # Parametri camera
        self.u_max = u_max
        self.v_max = v_max
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = u_max / 2
        self.c_y = v_max / 2
        
        # Parametri controllo
        self.lambda_ = lambda_
        self.max_velocity = max_velocity
    
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
    
    def compute_velocity(
        self,
        points_goal: Optional[npt.NDArray[np.float64]],
        points_current: Optional[npt.NDArray[np.float64]],
        depths: Optional[npt.NDArray[np.float64]] = None
    ) -> Optional[npt.NDArray[np.float64]]:
        """Calculate control velocity for IBVS"""
        if points_goal is None or points_current is None:
            return None
        
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
        
        return v_c.flatten()