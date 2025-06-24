import cv2
import socket
import struct
import pickle
import os

from dotenv import load_dotenv

load_dotenv()

cap = cv2.VideoCapture(0)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((os.getenv("CAMERA_TO_IP"), os.getenv("CAMERA_PORT")))

while True:
    ret, frame = cap.read()
    data = pickle.dumps(frame)
    message = struct.pack("Q", len(data)) + data
    client_socket.sendall(message)
