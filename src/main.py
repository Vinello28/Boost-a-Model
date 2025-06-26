import argparse
import time
import torch
import cv2
import logging
import os
import yaml
import traceback
import numpy as np

from PIL import Image
from models.vitvs.lib import VitVsLib
from pathlib import Path

from models.cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt
from models.cns.utils.perception import CameraIntrinsic
from models.cns.benchmark.stop_policy import SSIMStopPolicy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BAM_ROOT = "/workspace/Boost-a-Model/"
BAM_STORE = os.path.join(BAM_ROOT, ".store")
BAM_CONFIG = os.path.join(BAM_ROOT, ".config")
METHOD = "undefined"
RESULT_PATH= os.path.join(BAM_ROOT, METHOD, "results")
CONFIG_PATH = ""

_is_gui_enabled = True # Default GUI enabled, can be overridden by --no-gui flag

supported_methods = ["vit-vs", "cns", "test-vit-vs"]

def set_method(method):
    global METHOD
    global RESULT_PATH
    METHOD = method
    RESULT_PATH= os.path.join(BAM_ROOT, METHOD, "results")

def pad(img, multiple=14):
    w, h = img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple

    if (w, h) != (new_w, new_h):
        print(f"Padding image from {(w, h)} to {(new_w, new_h)}")

    padded = Image.new(img.mode, (new_w, new_h))
    padded.paste(img, (0, 0))
    return padded


def cns(reference, input_video, device, no_gui):
    """
    Esegue il benchmark CNS confrontando frame per frame due video.
    Salva i risultati in una cartella di output.
    """
    vis_opt = VisOpt.ALL if not no_gui else VisOpt.NO

    pipeline = CorrespondenceBasedPipeline(
        detector="AKAZE",
        ckpt_path="models/cns/checkpoints/cns_state_dict.pth",
        intrinsic=CameraIntrinsic.default(),
        device=device,
        ransac=True,
        vis=vis_opt
    )
    
    stop_policy = SSIMStopPolicy(
        waiting_time=2.0, # seconds to wait before stopping
        conduct_thresh=0.1 # error threshold
    )

    reference_cap = cv2.VideoCapture(reference)
    stream_cap = cv2.VideoCapture(input_video)
    results = []
    frame_idx = 0

    os.makedirs(RESULT_PATH, exist_ok=True)

    try:
        stop_policy.reset()
        while True:
            ref_ret, ref_frame = reference_cap.read()
            stream_ret, stream_frame = stream_cap.read()
            print("ref_ret:", ref_ret)
            if ref_frame is not None:
                print("ref_frame shape:", ref_frame.shape, "dtype:", ref_frame.dtype, "min:", ref_frame.min(), "max:", ref_frame.max())
                cv2.imwrite("debug_ref_frame.png", ref_frame)
            else:
                print("ref_frame is None")
                
            if not ref_ret or not stream_ret:
                break

            pipeline.set_target(ref_frame, dist_scale=1.0)
            
            vel, data, timing = pipeline.get_control_rate(stream_frame)
            if stop_policy(data, time.time()):
                break;
            
            # Salva risultati per ogni frame
            result = {
                "frame": frame_idx,
                "velocity": vel,
                "data": data,
                "timing": timing
            }
            results.append(result)

            # Salva immagine con keypoints CNS (se disponibili)
            if hasattr(pipeline, "frontend") and hasattr(pipeline.frontend, "last_vis"):
                out_path = os.path.join(
                    RESULT_PATH, f"cns_keypoints_{frame_idx:05d}.png")
                cv2.imwrite(out_path, pipeline.frontend.last_vis)

            frame_idx += 1

            if frame_idx % 10 == 0:
                print(f"[CNS] Processed {frame_idx} frames...")

        # Salva risultati in un file .npz o .json
        np.savez(os.path.join(
            RESULT_PATH, "cns_benchmark_results.npz"), results=results)
        print(
            f"[CNS] Benchmark completato. Risultati salvati in {RESULT_PATH}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception as e:
        logging.error("An error occurred:", exc_info=True)
    finally:
        reference_cap.release()
        stream_cap.release()

def vit_vs(reference, input_video, no_gui):
    config = {}

    # with open(os.path.join(BAM_CONFIG, "vit_vs_config.yaml"), 'r') as file:
    #     config = yaml.safe_load(file)
    # with open(os.path.expanduser("/workspace/src/models/vitvs/vitvs_config.yaml"), 'r') as file:
    #     config = yaml.safe_load(file)

    vit_vs = VitVsLib(config_path=config, gui=not no_gui)

    # Open the reference and stream videos
    reference_cap = cv2.VideoCapture(reference)
    stream_cap = cv2.VideoCapture(input_video)
    result = None
    ref_frame = None
    stream_frame = None

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
            goal_frame = pad(Image.fromarray(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)))
            current_frame = pad(Image.fromarray(cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)))
            
            result = vit_vs.process_frame_pair(goal_frame=pad(goal_frame), current_frame=pad(current_frame))
            logging.info(f"Result obtained")
            print(f"Result obtained")

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
        logging.error("An error occurred:", exc_info=True)  # full traceback in logs
    finally:
        # Release resources
        reference_cap.release()
        stream_cap.release()

def test_vit_vs(goal, current, device=None):
    print("🧪 ViT Visual Servoing System ")
    print("========================================")

    try:
        vitvs = VitVsLib(config_path=CONFIG_PATH if CONFIG_PATH else None, gui=_is_gui_enabled, device=device)

        logging.info(f"📋 Configuration params:")
        logging.info(f"   Model: {vitvs.model_type}")
        logging.info(f"   DINO input size: {vitvs.dino_input_size}")
        logging.info(f"   Num pairs: {vitvs.num_pairs}")
        logging.info(f"   Lambda: {vitvs.lambda_}")
        logging.info(f"   Camera: {vitvs.u_max}x{vitvs.v_max}")
        
        # Create results directory if it doesn't exist
        os.makedirs(RESULT_PATH, exist_ok=True)
        
        goal_name = Path(goal).stem
        current_name = Path(current).stem
        save_path = f"{RESULT_PATH}/keypoints_{goal_name}_vs_{current_name}.png"

        logging.info(f"🔍 Testing ViT Visual Servoing...")
        logging.info(f"   Goal: {goal}")
        logging.info(f"   Current: {current}")
        logging.info(f"   Metodo: Vision Transformer (DINOv2)")
        logging.info(f"   Output: {save_path}")

        goal_cap = cv2.VideoCapture(goal)
        current_cap = cv2.VideoCapture(current)
        result = None
        goal_frame = None
        current_frame = None

        while True:
            # Read next frame from each video
            # Proceed to next frame of goal video only if reference video
            # is "close enough" to the goal video
            if goal_frame is None or result is None or result.velocity < 1:
                goal_ret, goal_frame = goal_cap.read()
            current_ret, current_frame = current_cap.read()

            if not goal_ret:
                logging.info("End of goal video reached.")
            if not current_ret:
                logging.info("End of current video reached.")

            # Convert frames from BGR (OpenCV) to RGB (PIL)
            goal_frame = pad(Image.fromarray(cv2.cvtColor(goal_frame, cv2.COLOR_BGR2RGB)))
            current_frame = pad(Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)))

            # Test con ViT (sistema principale)
            result = vitvs.process_frame_pair(
                goal_frame, 
                current_frame, 
                save_path=save_path
            )

            print(f"Processed frame pair: {result}")

            if result:
                print(f"📊 Features detected: {result['num_features']}")
                print(f"🎯 Velocity:")
                velocity = result['velocity']
                print(f"   : vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}")
                print(f"   Rotate:   ωx={velocity[3]:.4f}, ωy={velocity[4]:.4f}, ωz={velocity[5]:.4f}")
                
                # Calcola norma velocità per valutazione
                velocity_norm = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2 + 
                            velocity[3]**2 + velocity[4]**2 + velocity[5]**2)**0.5
                print(f"📏 Velocity normalized: {velocity_norm:.4f}")
                
                # Info sulle coordinate dei punti (se disponibili)
                if 'goal_points' in result and 'current_points' in result:
                    print(f"📍 Goal points: {len(result['goal_points'])} punti")
                    print(f"📍 urrent points: {len(result['current_points'])} punti")
                
                # Conferma salvataggio immagine
                if os.path.exists(save_path):
                    print(f"💾 Keypoints saved in: {save_path}")
                else:
                    print(f"⚠️  Warning: Output file not found in {save_path}")
                    
            else:
                print("\n❌ FAILED! ViT Visual Servoing did not return a valid result.")

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception as e:
        logging.error("An error occurred:", exc_info=True)  # full traceback in logs
    finally:
        # Release resources
        goal_cap.release()
        current_cap.release()

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

    #TODO: Add feature to load from stdin -> this makes it real-time -> once available, make it default
    parser.add_argument(
        "--input",
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
 
    # parser.add_argument(
    #     "--`",
    #     type=str,
    #     default="",
    #     help="input video stream video path (default: ${BAM_ROOT}/data/stream.mp4). Could also be /dev/video0 for webcam input",
    # )

    args = parser.parse_args()

    _is_gui_enabled = not args.no_gui

    logging.info(f"Using method: {args.method}")
    logging.info(f"Reference: {args.reference}")
    logging.info(f"Input: {args.input}")
    logging.info(f"GUI: {_is_gui_enabled}")

    if args.method:
        set_method(args.method)

    # Set device via environment variable if specified
    if args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.replace('cuda:', '')
        logging.info(f"🎯 Forcing GPU device: {args.device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory > 20:  
            print("⚡ Fantasmagorical GPU detected - No memory limitations!")
        else:
            print("📊 Standard GPU detected")
    else:
        print("⚠️  CPU mode - crazy man, no GPU no powah!")

    # System initialization with config if provided
    if args.config:
        CONFIG_PATH = args.config
        print(f"📝 Using specified config: {CONFIG_PATH}")
    else:
        print("📝 Using default configuration")

    if args.method not in supported_methods:
        logging.error(f"Unsupported method: {args.method}. Supported methods: {supported_methods}")
        exit(-1)
    elif args.method == "cns":
        cns(reference=args.reference, input_video=args.input, device=args.device, no_gui=args.no_gui)
    elif args.method == "vit-vs":
        vit_vs(reference=args.reference, input_video=args.input, no_gui=args.no_gui)
    elif args.method == "test-vit-vs":
        test_vit_vs(goal=args.reference, current=args.input, device=args.device)
    else:
        logging.error(f"Method {args.method} is not implemented yet.")
        exit(-1)

    # Here you would typically load your configuration and start your application
    # For example:
    # config = load_config(args.config)
    # start_application(config)

