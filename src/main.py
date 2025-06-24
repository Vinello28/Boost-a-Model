from typing import Optional
import cv2
import logging
from util.exceptions import NoCameraError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_camera(
    capture_index: int = 0, from_socket: bool = False
) -> Optional[cv2.VideoCapture]:
    """
    Get camera stream, capture_index referes to index of the stream
    """
    if not from_socket:
        camera = cv2.VideoCapture(capture_index)

        if not camera.isOpened():
            logging.error("cannot open camera device")
            raise NoCameraError

        logging.info("camera ")
        return camera
    else:
        ## code to get stream from socket
        return None


def get_reference(path_to_ref: str):
    """
    Get reference to ground-truth file
    """
    try:
        file = open(path_to_ref, "r")
    except FileNotFoundError:
        logging.error(f"no {path_to_ref} found")
