import cv2

cap = cv2.VideoCapture('/tmp-video/stream-2025-06-24_14-51-22.mp4')

if not cap.isOpened():
    print("Failed to open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    cv2.imshow("Remote Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()