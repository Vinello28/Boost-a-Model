import argparse
import cv2
import logging
import os
import yaml

from PIL import Image
from models.vitvs.lib import VitVsLib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BAM_ROOT = "/workspace/bam"
BAM_STORE = os.path.join(BAM_ROOT, "store")
BAM_CONFIG = os.path.join(BAM_ROOT, "config")

supported_methods = ["vit-vs", "cns"]


def vit_vs(reference, input_video, no_gui):
    config = {}

    with open(os.path.join(BAM_CONFIG, "vit_vs_config.yaml"), 'r') as file:
        config = yaml.dump(config, file)

    vit_vs = VitVsLib(config_path=config, gui=not no_gui)

    # Open the reference and stream videos
    reference_cap = cv2.VideoCapture(reference)
    stream_cap = cv2.VideoCapture(input_video)
    result = None

    try:
        while True:
            # Read next frame from each video
            # Proceed to next frame of goal video only if reference video
            # is "close enough" to the goal video
            if ref_frame is None or result is None or result.velocity < 1:
                ref_ret, ref_frame = reference_cap.read()
            stream_ret, stream_frame = stream_cap.read()

            if not ref_ret or not stream_ret:
                break  # End of one of the videos

            # Convert frames from BGR (OpenCV) to RGB (PIL)
            goal_frame = Image.fromarray(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
            current_frame = Image.fromarray(cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB))
            
            result = vit_vs.process_frame_pair(goal_frame=goal_frame, current_frame=current_frame)

            if not no_gui:
                # Display the result in a window
                cv2.imshow("BAM - ViT-VS - Goal", goal_frame)
                cv2.imshow("BAM - ViT-VS - Reference", goal_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print(f"Processed frame pair: {result}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Release resources
        reference_cap.release()
        stream_cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BAM")

    parser.add_argument(
        "--method`",
        type=str,
        default="vit-vs",
        help="ViT-VS or CNS (vit-vs cns)",
    )

    parser.add_argument(
        "--reference`",
        type=str,
        default="/tmp-video/goal.mp4",
        help="reference video path (default: ${BAM_ROOT}/data/reference.mp4)",
    )

    #TODO: Add feature to load from stdin -> this makes it real-time -> once available, make it default
    parser.add_argument(
        "--input`",
        type=str,
        default="/tmp-video/stream.mp4",
        help="input video stream video path (default: ${BAM_ROOT}/data/stream.mp4). Could also be /dev/video0 for webcam input",
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Disable GUI. Use this flag to run without a graphical interface."
    )
 
    # parser.add_argument(
    #     "--`",
    #     type=str,
    #     default="",
    #     help="input video stream video path (default: ${BAM_ROOT}/data/stream.mp4). Could also be /dev/video0 for webcam input",
    # )

    args = parser.parse_args()
    logging.info(f"Using method: {args.method}")
    logging.info(f"Reference: {args.reference}")
    logging.info(f"Input: {args.input}")
    logging.info(f"GUI: {args.no_gui}")

    if args.method not in supported_methods:
        logging.error(f"Unsupported method: {args.method}. Supported methods: {supported_methods}")
        exit(-1)
    elif args.method == "vit-vs":
        vit_vs(reference=args.reference, input_video=args.input, no_gui=args.no_gui)
    else:
        logging.error(f"Method {args.method} is not implemented yet.")
        exit(-1)

    # Here you would typically load your configuration and start your application
    # For example:
    # config = load_config(args.config)
    # start_application(config)

