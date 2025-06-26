import argparse
import time
import torch
import cv2
import logging
import os
import numpy as np
import sys

from PIL import Image
from models.vitvs.lib import VitVsLib
from pathlib import Path

from util.data import Data
from models.cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt
from models.cns.utils.perception import CameraIntrinsic
from models.cns.benchmark.stop_policy import SSIMStopPolicy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RESULT_PATH = "results"
_is_gui_enabled = True  # Default GUI enabled, can be overridden by --no-gui flag

supported_methods = ["vit-vs", "cns", "test-vit-vs"]


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
        vis=vis_opt,
    )

    stop_policy = SSIMStopPolicy(
        waiting_time=2.0,  # seconds to wait before stopping
        conduct_thresh=0.1,  # error threshold
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

            if not ref_ret or not stream_ret:
                break

            pipeline.set_target(ref_frame, dist_scale=1.0)

            vel, data, timing = pipeline.get_control_rate(stream_frame)
            if stop_policy(data, time.time()):
                break

            # Salva risultati per ogni frame
            result = {
                "frame": frame_idx,
                "velocity": vel,
                "data": data,
                "timing": timing,
            }
            results.append(result)

            # Salva immagine con keypoints CNS (se disponibili)
            if hasattr(pipeline, "frontend") and hasattr(
                pipeline.frontend, "last_vis"
            ):
                out_path = os.path.join(
                    RESULT_PATH, f"cns_keypoints_{frame_idx:05d}.png"
                )
                cv2.imwrite(out_path, pipeline.frontend.last_vis)

            frame_idx += 1

            if frame_idx % 10 == 0:
                print(f"[CNS] Processed {frame_idx} frames...")

        # Salva risultati in un file .npz o .json
        np.savez(
            os.path.join(RESULT_PATH, "cns_benchmark_results.npz"),
            results=results,
        )
        print(f"[CNS] Benchmark completato. Risultati salvati in {RESULT_PATH}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception as e:
        logging.error("An error occurred:", exc_info=True)
    finally:
        reference_cap.release()
        stream_cap.release()


def test_vit_vs(data):
    gcap = None
    incap = None

    try:
        vitvs = VitVsLib(
            config_path=data.config_path or None,
            gui=data.state.is_gui_enabled or False,
            device=data.device or None,
        )

        logging.info(f"Model: {vitvs.model_type}")
        logging.info(f"DINO input size: {vitvs.dino_input_size}")
        logging.info(f"Num pairs: {vitvs.num_pairs}")
        logging.info(f"Lambda: {vitvs.lambda_}")
        logging.info(f"Camera: {vitvs.u_max}x{vitvs.v_max}")

        # Create results directory if it doesn't exist
        os.makedirs(data.result_path, exist_ok=True)

        # Check if config is right
        assert data.goal_path, "goal path not specified"
        assert data.input_path, "input path not specified"

        goal_name = Path(data.goal_path).stem
        current_name = Path(data.input_path).stem
        kp_out_path = (
            f"{data.result_path}/keypoints_{goal_name}_vs_{current_name}.png"
        )

        logging.info(f" Saving Keypoints into {kp_out_path}")

        gcap = cv2.VideoCapture(data.goal_path)
        incap = cv2.VideoCapture(data.input_path)

        result = None
        gf = None
        inf = None

        gt, gf = gcap.read()
        it, inf = incap.read()

        while gt and it:
            gt, gf = gcap.read()
            it, inf = incap.read()

            if not gt:
                logging.info("End of goal video reached.")
                break
            if not it:
                logging.info("End of current video reached.")
                break

            gf_pil = pad(Image.fromarray(cv2.cvtColor(gf, cv2.COLOR_BGR2RGB)))
            inf_pil = pad(Image.fromarray(cv2.cvtColor(inf, cv2.COLOR_BGR2RGB)))

            if data.state.is_gui_enabled:
                cv2.imshow(
                    "BAM - ViT-VS - Goal",
                    cv2.cvtColor(np.array(gf_pil), cv2.COLOR_RGB2BGR),
                )
                cv2.imshow(
                    "BAM - ViT-VS - Current",
                    cv2.cvtColor(np.array(inf_pil), cv2.COLOR_RGB2BGR),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            result = vitvs.process_frame_pair(gf_pil, inf_pil, save_path=kp_out_path)

            print(f"Processed frame pair: {result}")

            if result:
                velocity = result.velocity
                velocity_norm = np.linalg.norm(velocity)

                print(f"📊 Features detected: {result.num_features}")
                print("🎯 Velocity:")
                print(
                    f"   Translate: vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}"
                )
                print(
                    f"   Rotate:   ωx={velocity[3]:.4f}, ωy={velocity[4]:.4f}, ωz={velocity[5]:.4f}"
                )

                print(f"📏 Velocity normalized: {velocity_norm:.4f}")

                if result.points_goal and result.points_current:
                    print(f"📍 Goal points: {len(result.points_goal)} punti")
                    print(
                        f"📍 Current points: {len(result.points_current)} punti"
                    )

                if os.path.exists(kp_out_path):
                    print(f"💾 Keypoints saved in: {kp_out_path}")
                else:
                    print(
                        f"⚠️  Warning: Output file not found in {kp_out_path}"
                    )

            else:
                print(
                    "\n❌ FAILED! ViT Visual Servoing did not return a valid result."
                )

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception:
        logging.error("An error occurred:", exc_info=True)
    finally:
        if gcap:
            gcap.release()
        if incap:
            incap.release()
        cv2.destroyAllWindows()


def check_cuda() -> bool:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
        logging.info("CUDA is available for use")
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return True
    else:
        logging.warning("!!!CPU mode - performance may be affected!!!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run BAM")
    parser.add_argument(
        "--method",
        type=str,
        default="vit-vs",
        choices=supported_methods,
        help="Method to use for visual servoing.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="/tmp-video/goal.mp4",
        help="Reference video path.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/tmp-video/stream.mp4",
        help="Input video stream path.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for processing (e.g., 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    data = Data()
    data.cmd_args = args
    data.state.is_gui_enabled = not args.no_gui
    data.state.is_cuda_enabled = check_cuda()
    data.set_method(args.method)
    data.goal_path = args.reference
    data.input_path = args.input
    data.device = args.device
    data.config_path = args.config
    data.result_path = RESULT_PATH

    logging.info(f"Using method: {data.get_method()}")
    logging.info(f"Reference: {data.goal_path}")
    logging.info(f"Input: {data.input_path}")
    logging.info(f"GUI: {data.state.is_gui_enabled}")
    logging.info(f"CUDA: {data.state.is_cuda_enabled}")
    logging.info(f"Selected GPU device is: {data.device}")

    if data.config_path:
        logging.info(f"Using specified config: {data.config_path}")
    else:
        logging.info("Using default configuration")

    if data.device and "cuda" in data.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = data.device.replace("cuda:", "")

    match data.get_method():
        case "test-vit-vs":
            test_vit_vs(data)
        case "cns":
            cns(
                reference=args.reference,
                input_video=args.input,
                device=args.device,
                no_gui=args.no_gui,
            )
        case "vit-vs":
            logging.error(
                f"Method '{data.get_method()}' is not implemented yet."
            )
            sys.exit(1)
        case _:
            logging.error(
                f"Unsupported method: '{data.get_method()}'. Supported methods are: {supported_methods}"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
