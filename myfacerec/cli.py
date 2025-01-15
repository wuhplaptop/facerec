# myfacerec/cli.py

import argparse
import sys
from PIL import Image
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np

from .config import Config
from .facial_recognition import FacialRecognition
from .hooks import Hooks
from .combined_model import CombinedFacialRecognitionModel
from .data_store import JSONUserDataStore

def main():
    parser = argparse.ArgumentParser(
        prog="myfacerec",
        description="MyFaceRec: Modular and Extensible Facial Recognition Tool."
    )
    subparsers = parser.add_subparsers(dest="command")

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
    # Upload model (Optional)
    # -------------------------------------------------------------------------
    # [Implementation for upload-model can be added here if required.]

    # -------------------------------------------------------------------------
    # Download model (Optional)
    # -------------------------------------------------------------------------
    # [Implementation for download-model can be added here if required.]

    # -------------------------------------------------------------------------
    # Other existing commands (export-data, import-data, list-users, delete-user)
    # -------------------------------------------------------------------------
    # [Include existing commands here, omitted for brevity]

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command in ["register", "identify", "export-model", "import-model"]:
        # Initialize configuration
        config = Config(
            yolo_model_path=args.yolo_model,
            conf_threshold=args.conf if hasattr(args, 'conf') else 0.5,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # Handle commands
    if args.command == "register":
        # Initialize FacialRecognition
        fr = FacialRecognition(config)

        # If an existing combined model is provided, import it
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
        # Load the combined model
        if args.input_model and os.path.exists(args.input_model):
            fr = FacialRecognition.import_combined_model(args.input_model, config)
        else:
            fr = FacialRecognition(config)

        # Load image
        try:
            img = Image.open(args.image).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to open image {args.image}: {e}")
            sys.exit(1)

        # Identify
        results = fr.identify_user(img, threshold=args.threshold)
        for res in results:
            print(f"Face {res['face_id']}: User='{res['user_id']}', Similarity={res['similarity']:.2f}")

    elif args.command == "export-model":
        # Initialize FacialRecognition
        fr = FacialRecognition(config)

        # If an existing combined model is provided, import it
        if args.input_model and os.path.exists(args.input_model):
            fr = FacialRecognition.import_combined_model(args.input_model, config)

        # Export the combined model
        fr.export_combined_model(args.output_path)

    elif args.command == "import-model":
        # Import the combined model
        fr = FacialRecognition.import_combined_model(args.input_path, config)
        print(f"Combined model imported from {args.input_path}.")

    else:
        # Handle other commands here
        # [Implementation for other commands]
        pass

if __name__ == "__main__":
    main()
