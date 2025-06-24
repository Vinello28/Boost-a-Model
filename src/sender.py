import cv2
import socket
import struct
import pickle

cap = cv2.VideoCapture(0)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("0.0.0.0", 9999))

while True:
    ret, frame = cap.read()
    data = pickle.dumps(frame)
    message = struct.pack("Q", len(data)) + data
    client_socket.sendall(message)
