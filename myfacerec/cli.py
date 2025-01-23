# myfacerec/cli.py

import argparse
import sys
from PIL import Image
import shutil
import os
import torch

from .config import Config
from .facial_recognition import FacialRecognition
from .hooks import Hooks
from .combined_model import CombinedFacialRecognitionModel
from .data_store import JSONUserDataStore
from .training import train_yolo, train_facenet  # NEW

def main():
    parser = argparse.ArgumentParser(
        prog="myfacerec",
        description="MyFaceRec: Modular and Extensible Facial Recognition Tool."
    )
    subparsers = parser.add_subparsers(dest="command")

    # -------------------------------------------------------------------------
    # Common arguments for some subcommands
    # -------------------------------------------------------------------------
    # Because many commands might want to enable/disable pose:
    parser.add_argument("--enable-pose", action="store_true", help="Enable head pose estimation.")

    # -------------------------------------------------------------------------
    # Register command
    # -------------------------------------------------------------------------
    reg_parser = subparsers.add_parser("register", help="Register a user.")
    reg_parser.add_argument("--user", required=True, help="User ID (unique).")
    reg_parser.add_argument("--images", nargs="+", required=True, help="Paths to one or more face images.")
    reg_parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    reg_parser.add_argument("--yolo-model", default=None, help="Path or URL to YOLO model (.pt). If not set, default model is used.")
    reg_parser.add_argument("--output-model", default=None, help="Path to save the combined model after registration.")

    # -------------------------------------------------------------------------
    # Identify command
    # -------------------------------------------------------------------------
    id_parser = subparsers.add_parser("identify", help="Identify faces in an image.")
    id_parser.add_argument("--image", required=True, help="Path to input image.")
    id_parser.add_argument("--threshold", type=float, default=0.6, help="Recognition threshold.")
    id_parser.add_argument("--yolo-model", default=None, help="Path or URL to YOLO model (.pt). If not set, default model is used.")
    id_parser.add_argument("--input-model", default=None, help="Path to the combined model (.pt).")

    # -------------------------------------------------------------------------
    # Export model
    # -------------------------------------------------------------------------
    export_model_parser = subparsers.add_parser("export-model", help="Export the combined facial recognition model.")
    export_model_parser.add_argument("--output-path", required=True, help="Path to save the exported model (.pt).")
    export_model_parser.add_argument("--yolo-model", default=None, help="Path or URL to YOLO model (.pt). If not set, default model is used.")
    export_model_parser.add_argument("--input-model", default=None, help="Path to the existing combined model (.pt). If not set, a new model is initialized.")

    # -------------------------------------------------------------------------
    # Import model
    # -------------------------------------------------------------------------
    import_model_parser = subparsers.add_parser("import-model", help="Import a combined facial recognition model.")
    import_model_parser.add_argument("--input-path", required=True, help="Path to the combined model (.pt).")
    import_model_parser.add_argument("--yolo-model", default=None, help="Path or URL to YOLO model (.pt). If not set, default model is used.")

    # -------------------------------------------------------------------------
    # Train command (new)
    # -------------------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train or fine-tune YOLO or FaceNet.")
    train_parser.add_argument("--model-type", choices=["yolo", "facenet"], default="facenet",
                             help="Which model to train/fine-tune.")
    train_parser.add_argument("--data-dir", required=True, help="Path to the dataset for training.")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    train_parser.add_argument("--save-path", default="trained_model.pt", help="Where to save the trained model weights.")
    # (Add more training args as needed)

    # -------------------------------------------------------------------------
    # Parse
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Build config object (common for all commands)
    config = Config(
        yolo_model_path=args.yolo_model if hasattr(args, "yolo_model") else None,
        conf_threshold=args.conf if hasattr(args, "conf") else 0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_pose_estimation=args.enable_pose  # pass the pose flag
    )

    # Handle commands
    if args.command == "register":
        fr = FacialRecognition(config)

        # If an existing combined model is provided (output_model) and it exists, import it
        if args.output_model and os.path.exists(args.output_model):
            fr = FacialRecognition.import_combined_model(args.output_model, config)

        image_paths = args.images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[Error] Failed to open image {img_path}: {e}")

        if not images:
            print("[Error] No valid images to register.")
            sys.exit(1)

        message = fr.register_user(args.user, images)
        print(message)

        # Save the combined model if output path is provided
        if args.output_model:
            fr.export_combined_model(args.output_model)

    elif args.command == "identify":
        if args.input_model and os.path.exists(args.input_model):
            fr = FacialRecognition.import_combined_model(args.input_model, config)
        else:
            fr = FacialRecognition(config)

        try:
            img = Image.open(args.image).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to open image {args.image}: {e}")
            sys.exit(1)

        results = fr.identify_user(img, threshold=args.threshold)
        for res in results:
            face_id = res['face_id']
            user_id = res['user_id']
            sim = res['similarity']
            box = res['box']
            pose = res['pose']
            print(f"Face {face_id}: User='{user_id}', Similarity={sim:.2f}, Box={box}, Pose={pose}")

    elif args.command == "export-model":
        fr = FacialRecognition(config)
        if args.input_model and os.path.exists(args.input_model):
            fr = FacialRecognition.import_combined_model(args.input_model, config)
        fr.export_combined_model(args.output_path)

    elif args.command == "import-model":
        fr = FacialRecognition.import_combined_model(args.input_path, config)
        print(f"Combined model imported from {args.input_path}.")

    elif args.command == "train":
        if args.model_type == "yolo":
            train_yolo(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)
        else:
            train_facenet(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
