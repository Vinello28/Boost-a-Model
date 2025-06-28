import requests

url = "http://localhost:8000/analyze"
goal_video_path = "goal.mp4"  # Sostituisci con il tuo file
current_video_path = "current.mp4"  # Sostituisci con il tuo file

goal_frame_idx = 0  # Cambia se vuoi un altro frame
current_frame_idx = 0

with open(goal_video_path, "rb") as f1, open(current_video_path, "rb") as f2:
    files = {
        "goal_video": ("goal.mp4", f1, "video/mp4"),
        "current_video": ("current.mp4", f2, "video/mp4"),
    }
    params = {"goal_frame_idx": goal_frame_idx, "current_frame_idx": current_frame_idx}
    response = requests.post(url, files=files, params=params)

print("Status code:", response.status_code)
try:
    print("Risposta JSON:", response.json())
except Exception:
    print("Risposta:", response.text)
