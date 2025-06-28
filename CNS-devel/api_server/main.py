import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn

# Import CNS pipeline (assumendo che CNS-devel sia nel PYTHONPATH)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cns_external_images import run_cns_with_external_images

app = FastAPI()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEVICE = None  # Device globale

@app.post("/analyze")
def analyze(
    goal_video: UploadFile = File(...),
    current_video: UploadFile = File(...),
    goal_frame_idx: int = 0,
    current_frame_idx: int = 0
):
    import numpy as np
    import cv2
    import tempfile
    # Salva i video temporaneamente
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f_goal:
        f_goal.write(goal_video.file.read())
        goal_path = f_goal.name
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f_curr:
        f_curr.write(current_video.file.read())
        curr_path = f_curr.name
    try:
        # Estrai i frame desiderati
        def estrai_frame(video_path, frame_idx):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise HTTPException(status_code=400, detail=f"Impossibile estrarre il frame {frame_idx} da {video_path}")
            return frame
        goal_img = estrai_frame(goal_path, goal_frame_idx)
        current_img = estrai_frame(curr_path, current_frame_idx)
        if goal_img is None or current_img is None:
            raise HTTPException(status_code=400, detail="Immagine non valida")
        # Esegui CNS pipeline sui frame estratti
        vel, data, timing = run_cns_with_external_images(goal_img=goal_img, current_img=current_img, device=DEVICE)
        if vel is None:
            raise HTTPException(status_code=400, detail="CNS pipeline failed")
        return JSONResponse({
            "velocity": vel.tolist() if hasattr(vel, 'tolist') else vel,
            "timing": timing,
            "data": str(data)  # serializzazione semplice, puoi migliorare
        })
    finally:
        # Rimuovi i file temporanei
        if os.path.exists(goal_path):
            os.remove(goal_path)
        if os.path.exists(curr_path):
            os.remove(curr_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help="Device su cui eseguire CNS: 'cuda:0' o 'cpu'")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    print(f"[INFO] CNS device impostato su: {DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
