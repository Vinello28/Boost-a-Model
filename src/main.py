import argparse
import time
import torch
import cv2
import logging
import os
import numpy as np

from PIL import Image
from models.vitvs.lib import VitVsLib
from pathlib import Path

from util.data import Data

from util.data import Data
from models.cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt
from models.cns.utils.perception import CameraIntrinsic
from models.cns.benchmark.stop_policy import SSIMStopPolicy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            print("ref_ret:", ref_ret)
            if ref_frame is not None:
                print(
                    "ref_frame shape:",
                    ref_frame.shape,
                    "dtype:",
                    ref_frame.dtype,
                    "min:",
                    ref_frame.min(),
                    "max:",
                    ref_frame.max(),
                )
                cv2.imwrite("debug_ref_frame.png", ref_frame)
            else:
                print("ref_frame is None")

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
            if hasattr(pipeline, "frontend") and hasattr(pipeline.frontend, "last_vis"):
                out_path = os.path.join(
                    RESULT_PATH, f"cns_keypoints_{frame_idx:05d}.png"
                )
                cv2.imwrite(out_path, pipeline.frontend.last_vis)

            frame_idx += 1

            if frame_idx % 10 == 0:
                print(f"[CNS] Processed {frame_idx} frames...")

        # Salva risultati in un file .npz o .json
        np.savez(
            os.path.join(RESULT_PATH, "cns_benchmark_results.npz"), results=results
        )
        print(f"[CNS] Benchmark completato. Risultati salvati in {RESULT_PATH}")

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception as e:
        logging.error("An error occurred:", exc_info=True)
    finally:
        reference_cap.release()
        stream_cap.release()


# def vit_vs(goal_path: str, input_path: str, gui: bool = False):
#     # WARNING: This always uses default config!
#     vit_vs = VitVsLib(gui=gui)
#
#     # Open the reference and stream videos
#     reference_cap = cv2.VideoCapture(goal_path)
#     stream_cap = cv2.VideoCapture(input_path)
#
#     result = None
#     ref_ret, ref_frame = reference_cap.read()
#     stream_ret, stream_frame = stream_cap.read()
#
#     try:
#         while True:
#             # Read next frame from each video
#             # Proceed to next frame of goal video only if reference video
#             # is "close enough" to the goal video
#             if ref_frame is None or result is None or result.velocity < 1:
#                 ref_ret, ref_frame = reference_cap.read()
#             stream_ret, stream_frame = stream_cap.read()
#
#             if not ref_ret or not stream_ret:
#                 break  # End of one of the videos
#
#             # Convert frames from BGR (OpenCV) to RGB (PIL)
#             goal_frame = pad(
#                 Image.fromarray(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
#             )
#             current_frame = pad(
#                 Image.fromarray(cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB))
#             )
#
#             result = vit_vs.process_frame_pair(
#                 goal_frame=pad(goal_frame), current_frame=pad(current_frame)
#             )  # type: ignore
#             logging.info(f"Result obtained")
#             print(f"Result obtained")
#
#             if not gui:
#                 # Display the result in a window
#                 cv2.imshow("BAM - ViT-VS - Goal", goal_frame)  # type: ignore
#                 cv2.imshow("BAM - ViT-VS - Reference", goal_frame)  # type: ignore
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break
#
#             print(f"Processed frame pair: {result}")
#     except KeyboardInterrupt:
#         logging.info("Interrupted by user, exiting...")
#     except Exception as e:
#         logging.error("An error occurred:", exc_info=True)  # full traceback in logs
#     finally:
#         # Release resources
#         reference_cap.release()
#         stream_cap.release()


def test_vit_vs():
    gcap = None
    incap = None
    error_count = 0
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
        kp_out_path = f"{data.result_path}/keypoints_{goal_name}_vs_{current_name}.png"

        logging.info(f" Saving Keypoints into {kp_out_path}")

        gcap = cv2.VideoCapture(data.goal_path)
        incap = cv2.VideoCapture(data.input_path)

        result = None
        gf = None
        inf = None
        # gt = goalt ret, it is used to check if a frame was sucessfully captured
        # it = input ret, same as above
        # if gt or it are False, it means one of the two or both have reached EOF
        #
        # gf = goal frame, content of the frame
        # ing = input frame, same as above
        gt, gf = gcap.read()
        it, inf = incap.read()

        # continue until one of the two reaches EOF
        # NOTE: not a best practice but for PoC purposes is just enough
        while gt and it:
            # Read next frame from each video
            # Proceed to next frame of goal video only if reference video
            # is "close enough" to the goal video

            gt, gf = gcap.read()
            it, inf = incap.read()

            if not gt:
                logging.info("End of goal video reached.")
            if not it:
                logging.info("End of current video reached.")
            if not gt or not it:
                break

            # NOTE: here we pad the frames so that they are a multiple of 14 (required by dino i think)
            # NOTE: might be a region of interest in case of something not working correctly
            gf = Image.fromarray(cv2.cvtColor(gf, cv2.COLOR_BGR2RGB))
            inf = Image.fromarray(cv2.cvtColor(inf, cv2.COLOR_BGR2RGB))

            if data.state.is_gui_enabled:
                # Display the goal and current frames in a window
                cv2.imshow("BAM - ViT-VS - Goal", np.array(gf))
                cv2.imshow("BAM - ViT-VS - Current", np.array(inf))
            result = vitvs.process_frame_pair(gf, inf, save_path=kp_out_path)

            error_count = 0  # Reset error count on successful processing

            if result:
                logging.info("results are valid")
                # velocity = result.velocity
                # velocity_norm = (
                #     velocity[0] ** 2
                #     + velocity[1] ** 2
                #     + velocity[2] ** 2
                #     + velocity[3] ** 2
                #     + velocity[4] ** 2
                #     + velocity[5] ** 2
                # ) ** 0.5

                print(f"📊  Features detected:      {result.num_features}")
                print(f"📊  Input Points detected:  {result.points_current}")
                print(f"📊  Goal Points detected:   {result.points_goal}")
                print(f"📊  Velocity to be applied: {result.velocity}")

            else:
                logging.error("❌ result content is None (somehow)")

    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting...")
    except Exception:
        logging.error("An error occurred:", exc_info=True)  # full traceback in logs
    finally:
        # Release resources if ever allocated
        if isinstance(gcap, cv2.VideoCapture):
            gcap.release()
        if isinstance(incap, cv2.VideoCapture):
            incap.release()


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
        "--no-gui",
        action="store_true",
        default=False,
        help="Disable GUI. Use this flag to run without a graphical interface.",
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
    data = Data()
    args = parser.parse_args()
    data.cmd_args = args

    data.state.is_gui_enabled = not args.no_gui
    data.state.is_cuda_enabled = check_cuda()
    data.set_method(args.method)
    data.goal_path = args.reference
    data.input_path = args.input
    data.device = args.device
    data.config_path = args.config

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

    match data.get_method():
        case "test-vit-vs":
            test_vit_vs()
        case "cns":
            cns(
                reference=args.reference,
                input_video=args.input,
                device=args.device,
                no_gui=args.no_gui,
            )
        case _:
            logging.error("Method not found")

    if data.get_method() not in supported_methods:
        logging.error(
            f"Unsupported method: {data.get_method()}. Supported methods: {supported_methods}"
        )
        exit(-1)
    else:
        logging.error(f"Method {data.get_method()} is not implemented yet.")
        exit(-1)
