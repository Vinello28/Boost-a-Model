import argparse
import cv2
import logging
import os
import yaml
import traceback
import numpy as np

from PIL import Image
from models.vitvs.lib import VitVsLib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BAM_ROOT = "/workspace/Boost-a-Model/"
BAM_STORE = os.path.join(BAM_ROOT, "store")
BAM_CONFIG = os.path.join(BAM_ROOT, "config")

supported_methods = ["vit-vs", "cns"]


def pad(img, multiple=14):
    w, h = img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple

    if (w, h) != (new_w, new_h):
        print(f"Padding image from {(w, h)} to {(new_w, new_h)}")

    padded = Image.new(img.mode, (new_w, new_h))
    padded.paste(img, (0, 0))
    return padded

# def vit_vs(reference, input_video, no_gui):
#     config = {}

#     # with open(os.path.join(BAM_CONFIG, "vit_vs_config.yaml"), 'r') as file:
#     #     config = yaml.safe_load(file)
#     # with open(os.path.expanduser("/workspace/src/models/vitvs/vitvs_config.yaml"), 'r') as file:
#     #     config = yaml.safe_load(file)

#     vit_vs = VitVsLib(config_path=config, gui=not no_gui)

#     # Open the reference and stream videos
#     reference_cap = cv2.VideoCapture(reference)
#     stream_cap = cv2.VideoCapture(input_video)
#     result = None
#     ref_frame = None
#     stream_frame = None

#     try:
#         while True:
#             # Read next frame from each video
#             # Proceed to next frame of goal video only if reference video
#             # is "close enough" to the goal video
#             if ref_frame is None or result is None or result.velocity < 1:
#                 ref_ret, ref_frame = reference_cap.read()
#             stream_ret, stream_frame = stream_cap.read()

#             if not ref_ret or not stream_ret:
#                 break  # End of one of the videos

#             # Convert frames from BGR (OpenCV) to RGB (PIL)
#             goal_frame = pad(Image.fromarray(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)))
#             current_frame = pad(Image.fromarray(cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)))
            
#             result = vit_vs.process_frame_pair(goal_frame=pad(goal_frame), current_frame=pad(current_frame))
#             logging.info(f"Result obtained")
#             print(f"Result obtained")

#             if not no_gui:
#                 # Display the result in a window
#                 cv2.imshow("BAM - ViT-VS - Goal", goal_frame)
#                 cv2.imshow("BAM - ViT-VS - Reference", goal_frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             print(f"Processed frame pair: {result}")
#     except KeyboardInterrupt:
#         logging.info("Interrupted by user, exiting...")
#     except Exception as e:
#         logging.error("An error occurred:", exc_info=True)  # full traceback in logs
#     finally:
#         # Release resources
#         reference_cap.release()
#         stream_cap.release()

def test_vit_features(reference, input, device=None):
    # Set device via environment variable if specified
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device.replace('cuda:', '')
        print(f"🎯 Forcing GPU device: {device}")
    
    print("🧪 Test ViT Visual Servoing System")
    print("========================================")
    print("Sistema esclusivamente basato su Vision Transformer")
    
    # Check GPU info
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
    
    try:
        # System initialization with config if provided
        config_path = None
        if hasattr(args, 'config') and args.config:
            config_path = args.config
            print(f"📝 Using specified config: {config_path}")
        elif os.path.exists("vitvs_config.yaml"):
            config_path = "vitvs_config.yaml"
            print(f"📝 Using default config: {config_path}")
        else:
            print("📝 Using default parameters (no config file found)")
            
        vitvs = VitVsLib(config_path=config_path)
        print("✅ Sistema ViT-VS inizializzato")
        
        # Show loaded configuration
        if config_path:
            print(f"📋 Parametri configurazione:")
            print(f"   Model: {vitvs.model_type}")
            print(f"   DINO input size: {vitvs.dino_input_size}")
            print(f"   Num pairs: {vitvs.num_pairs}")
            print(f"   Lambda: {vitvs.lambda_}")
            print(f"   Camera: {vitvs.u_max}x{vitvs.v_max}")
        
        # Test with images from dataset
        goal_path = "dataset_small/comandovitruviano.jpeg"
        current_path = "dataset_small/curr3.jpeg"
        
        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate save path for keypoints visualization
        from pathlib import Path
        goal_name = Path(goal_path).stem
        current_name = Path(current_path).stem
        save_path = f"{results_dir}/keypoints_{goal_name}_vs_{current_name}.png"
        
        print(f"\n🔍 Testing ViT Visual Servoing...")
        print(f"   Goal: {goal_path}")
        print(f"   Current: {current_path}")
        print(f"   Metodo: Vision Transformer (DINOv2)")
        print(f"   Output: {save_path}")
        
        # Test con ViT (sistema principale)
        result = vitvs.process_image_pair(
            goal_path, 
            current_path, 
            visualize=True,
            save_path=save_path
        )
        
        if result:
            print(f"\n✅ SUCCESS! ViT Visual Servoing funziona!")
            print(f"📊 Features rilevate: {result['num_features']}")
            print(f"🎯 Velocità calcolata:")
            velocity = result['velocity']
            print(f"   Traslazione: vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f}")
            print(f"   Rotazione:   ωx={velocity[3]:.4f}, ωy={velocity[4]:.4f}, ωz={velocity[5]:.4f}")
            
            # Calcola norma velocità per valutazione
            velocity_norm = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2 + 
                           velocity[3]**2 + velocity[4]**2 + velocity[5]**2)**0.5
            print(f"📏 Norma velocità: {velocity_norm:.4f}")
            
            # Info sulle coordinate dei punti (se disponibili)
            if 'goal_points' in result and 'current_points' in result:
                print(f"📍 Coordinate goal points: {len(result['goal_points'])} punti")
                print(f"📍 Coordinate current points: {len(result['current_points'])} punti")
                
            # Info sulla similarità media
            if 'velocity_norm' in result:
                print(f"🎯 Velocità normalizzata: {result['velocity_norm']:.4f}")
            
            # Conferma salvataggio immagine
            if os.path.exists(save_path):
                print(f"💾 Keypoints salvati in: {save_path}")
            else:
                print(f"⚠️  Attenzione: File di output non trovato in {save_path}")
                
        else:
            print("\n❌ FAILED! ViT Visual Servoing non funziona")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        traceback.print_exc()


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

