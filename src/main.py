import argparse
import json
import torch
import cv2
import logging
import os

from PIL import Image
from models.vitvs.lib import VitVsLib

from util.data import Data

from models.baby_cns.lib import CnsLib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def pad(img: Image.Image, multiple: int = 14) -> Image.Image:
    w, h = img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple

    if (w, h) != (new_w, new_h):
        print(f"Padding image from {(w, h)} to {(new_w, new_h)}")

    padded = Image.new(img.mode, (new_w, new_h))
    padded.paste(img, (0, 0))
    return padded


def cns():
    """
    Esegue il benchmark CNS confrontando frame per frame due video.
    Salva i risultati in una cartella di output.
    """
    cns = CnsLib(data.device)

    goal_cap = cv2.VideoCapture(data.goal_path)
    input_cap = cv2.VideoCapture(data.input_path)

    results = []
    
    frame_idx = 0

    os.makedirs(data.result_path, exist_ok=True)

    try:
        while True:
            ref_ret, ref_frame = goal_cap.read()
            stream_ret, stream_frame = input_cap.read()

            if not ref_ret or not stream_ret:
                break

            cns.set_target(ref_frame)

            logging.info("[INFO] Processing frames with CNS pipeline...")
            vel, data_p, timing = cns.get_control_rate(stream_frame)
                     
            if hasattr(data_p, 'keys'):
                print(f"[INFO] Data keys: {list(data.keys())}")
                
                result = {
                    "frame": cns.convert_to_json_serializable(frame_idx),
                    "velocity":cns.convert_to_json_serializable(vel),
                    "data": cns.convert_to_json_serializable(data_p),
                    "timing": cns.convert_to_json_serializable(timing),
                }
                results.append(result)
          
            frame_idx += 1

            if frame_idx % 10 == 0:
                logging.info(f"[CNS] Processed {frame_idx} frames...")   
                
        # Save results to JSON file
        output_file = "../bam-results/cns/results/cns_external_results.json"
        cns.save_results(results, output_file)

        logging.info(
            f"[CNS] Benchmark completato. Risultati salvati in {data.result_path}"
        )

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception:
        logging.error("An error occurred:", exc_info=True)
    finally:
        goal_cap.release()
        input_cap.release()


def vitvs():
    gcap = None
    incap = None

    data.reset_time_points()

    try:
        vitvs = VitVsLib(
            config_path=data.config_path or None,
            gui=data.state.is_gui_enabled or False,
            device=data.device or None,
            metrics_save_path=os.path.join(
                data.result_path, f"metrics-{data.str_time_start}.json"
            ),
        )

        logging.info(f"Model: {vitvs.config.model_type}")
        logging.info(f"DINO input size: {vitvs.config.dino_input_size}")
        logging.info(f"Num pairs: {vitvs.config.num_pairs}")
        logging.info(f"Lambda: {vitvs.config.lambda_}")
        logging.info(f"Camera: {vitvs.config.u_max}x{vitvs.config.v_max}")

        # Create results directory if it doesn't exist
        os.makedirs(data.result_path, exist_ok=True)

        # Check if config is right
        assert data.goal_path, "goal path not specified"
        assert data.input_path, "input path not specified"

        kp_out_path = f"{data.result_path}/{data.progress}.png"

        logging.info(f" Saving Keypoints into {kp_out_path}")

        gcap = cv2.VideoCapture(data.goal_path)
        incap = cv2.VideoCapture(data.input_path)

        gf = None
        inf = None

        if data.still_frame_number >= -1:
            data.gf_position = data.still_frame_number
            gf = get_frame_at(data.goal_path, data.still_frame_number)
            gt = True
        else:
            gt, gf = gcap.read()

        it, inf = incap.read()

        
        results = []

        # continue until one of the two reaches EOF
        # NOTE: not a best practice but for PoC purposes is just enough
        while gt and it:
            # Read next frame from each video
            # Proceed to next frame of goal video only if reference video
            # is "close enough" to the goal video
            
            if data.still_frame_number < 0:
                gt, gf = gcap.read()
                data.gf_position += 1
            it, inf = incap.read()

            data.progress += 1
            data.inf_position += 1
            kp_out_file_name = os.path.join(kp_out_path, f"{data.progress}.png")

            if data.max_frames > 0 and data.progress >= data.max_frames:
                logging.info(
                    f"Reached maximum number of frames to process: {data.max_frames}"
                )
                break

            if not gt:
                logging.info("End of goal video reached.")
            if not it:
                logging.info("End of current video reached.")
            if not gt or not it:
                break

            # NOTE: here we WERE padding the frames so that they are a multiple of 14 (required by dino i think)
            # might be a region of interest in case of something not working correctly
            # it is not being done anymore and it works (somehow)
            gf = Image.fromarray(cv2.cvtColor(gf, cv2.COLOR_BGR2RGB))
            inf = Image.fromarray(cv2.cvtColor(inf, cv2.COLOR_BGR2RGB))

            vitvs.process_frame_pair(gf, inf, save_path=kp_out_file_name)

        if data.state.is_metrics_enabled:
            vitvs.save_metrics()

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception:
        logging.error("An error occurred:", exc_info=True)  # full traceback in logs
    finally:
        # Release resources if ever allocated
        logging.info(f"avarage time it took to process frames {data.get_avg_time()}")
        if isinstance(gcap, cv2.VideoCapture) and gcap is not None:
            gcap.release()
        else:
            logging.warning("strangely gcap was None or not a instance of VideoCapture")
        if isinstance(incap, cv2.VideoCapture) and incap is not None:
            incap.release()
        else:
            logging.warning(
                "strangely incap was None or not a instance of VideoCapture"
            )

def get_frame_at(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        raise ValueError(f"Requested frame {frame_number} exceeds total frames {total_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Failed to read frame {frame_number}")
    return frame

def check_cuda() -> bool:
    # Check for cuda availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info("CUDA is available for use")
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

        if gpu_memory > 20:
            logging.info(
                "Fantasmagorical GPU detected - Close to no memory limitations! (with great power comes great responsability)"
            )
        else:
            logging.info("Standard GPU detected")
        return True
    else:
        logging.warning("!!!CPU mode - crazy man, no GPU no powah!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BAM")

    parser.add_argument(
        "--method",
        type=str,
        default="vit-vs",
        help="ViT-VS or CNS (vit-vs cns)",
    )

    parser.add_argument(
        "--reference",
        type=str,
        default="/tmp-video/goal.mp4",
        help="reference video path (default: ${BAM_ROOT}/data/reference.mp4)",
    )

    # TODO: Add feature to load from stdin -> this makes it real-time -> once available, make it default
    parser.add_argument(
        "--input",
        type=str,
        default="/tmp-video/stream.mp4",
        help="input video stream video path (default: ${BAM_ROOT}/data/stream.mp4). Could also be /dev/video0 for webcam input",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu, cuda:0 or cuda:2 (default: cpu). Set the device to use for processing. If using CUDA, ensure the correct GPU is specified.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file (optional). If not provided, default parameters will be used.",
    )

    #### ACTIONS

    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Disable GUI. Use this flag to run without a graphical interface.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Enable rendering of correspondences for visualization (this makes runing slower)",
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        default=False,
        help="Enable metrics",
    )
    
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process (default: -1). This is useful for testing purposes. Set to -1 to process all frames.",
    )

    parser.add_argument(
        "--reference-still-frame",
        type=int,
        default=-1,
        help="Frame number to use as reference still frame (default: None). This is useful for testing purposes." 
    )
    
    

    data = Data()
    args = parser.parse_args()
    data.cmd_args = args

    data.state.is_gui_enabled = args.gui
    data.state.is_cuda_enabled = check_cuda()
    data.state.is_render_enabled = args.render

    data.set_method(args.method)
    data.goal_path = args.reference
    data.input_path = args.input
    data.device = args.device
    data.config_path = args.config
    data.state.is_metrics_enabled = args.metrics
    data.max_frames = args.max_frames

    logging.info(f"Using method: {data.get_method()}")
    logging.info(f"Reference: {data.goal_path}")
    logging.info(f"Input: {data.input_path}")
    logging.info(f"GUI: {data.state.is_gui_enabled}")
    logging.info(f"CUDA: {data.state.is_cuda_enabled}")
    logging.info(f"Selected GPU device is: {data.device}. Checking availability")
    if data.config_path:
        logging.info(f"Using specified config: {data.config_path}")
    else:
        logging.info("Using default configuration")

    logging.info(f"Starting method: {data.get_method()}")

    # Set device via environment variable if specified
    if data.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = data.device.replace("cuda:", "")

    # NOTE: THIS IS WERE YOU SHOULD SPECIFY OTHER METHODS
    supported_methods = {"vitvs": vitvs, "cns": cns}

    method_func = supported_methods.get(data.get_method())
    if method_func:
        method_func()
    else:
        logging.error("Provided method is not supported or not implemented yet.")
        exit(os.EX_USAGE)

    # match data.get_method():
    #     case "vitvs":
    #         vitvs()
    #     case "cns":
    #         cns(
    #             reference=args.reference,
    #             input_video=args.input,
    #             device=args.device,
    #             no_gui=args.no_gui,
    #         )
    #     case _:
