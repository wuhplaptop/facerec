# myfacerec/cli.py

import argparse
import sys
from PIL import Image
import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from .facial_recognition import FacialRecognition
from .hooks import Hooks
from ultralytics import YOLO
import torch
from facenet_pytorch import InceptionResnetV1
from .combined_model import CombinedFacialRecognitionModel  # Import the combined model
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        prog="rolo-rec",
        description="Rolo-Rec: Modular and Extensible Facial Recognition Tool."
    )
    subparsers = parser.add_subparsers(dest="command")

    # -------------------------------------------------------------------------
    # Register command
    # -------------------------------------------------------------------------
    reg_parser = subparsers.add_parser("register", help="Register a user.")
    reg_parser.add_argument("--user", required=True, help="User ID (unique).")
    reg_parser.add_argument("--images", nargs="+", required=True, help="Paths to one or more face images.")
    reg_parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    reg_parser.add_argument("--model-path", default=None,
                            help="Path or URL to custom YOLO model (.pt). If not set, default model is used.")
    reg_parser.add_argument("--detector", default=None,
                            help="Name of the detector plugin to use.")
    reg_parser.add_argument("--embedder", default=None,
                            help="Name of the embedder plugin to use.")
    reg_parser.add_argument("--parallelism", type=int, default=4,
                            help="Number of parallel threads for processing.")
    reg_parser.add_argument("--combined-model-path", default=None,
                            help="Path to a combined YOLO and Facenet .pt model.")

    # -------------------------------------------------------------------------
    # Identify command
    # -------------------------------------------------------------------------
    id_parser = subparsers.add_parser("identify", help="Identify faces in an image.")
    id_parser.add_argument("--image", required=True, help="Path to input image.")
    id_parser.add_argument("--threshold", type=float, default=0.6, help="Recognition threshold.")
    id_parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    id_parser.add_argument("--model-path", default=None,
                           help="Path or URL to custom YOLO model (.pt). If not set, default model is used.")
    id_parser.add_argument("--detector", default=None,
                           help="Name of the detector plugin to use.")
    id_parser.add_argument("--embedder", default=None,
                           help="Name of the embedder plugin to use.")
    id_parser.add_argument("--combined-model-path", default=None,
                           help="Path to a combined YOLO and Facenet .pt model.")

    # -------------------------------------------------------------------------
    # Export data
    # -------------------------------------------------------------------------
    export_parser = subparsers.add_parser("export-data", help="Export the face data (JSON) to a specified file.")
    export_parser.add_argument("--output", required=True, help="Path to save the exported data.")

    # -------------------------------------------------------------------------
    # Import data
    # -------------------------------------------------------------------------
    import_parser = subparsers.add_parser("import-data", help="Import face data from a JSON file.")
    import_parser.add_argument("--file", required=True, help="Path to the JSON file to import.")

    # -------------------------------------------------------------------------
    # List users
    # -------------------------------------------------------------------------
    list_parser = subparsers.add_parser("list-users", help="List all registered users.")
    list_parser.add_argument("--combined-model-path", default=None,
                             help="Path to a combined YOLO and Facenet .pt model.")

    # -------------------------------------------------------------------------
    # Delete user
    # -------------------------------------------------------------------------
    delete_parser = subparsers.add_parser("delete-user", help="Delete a registered user.")
    delete_parser.add_argument("--user", required=True, help="User ID to delete.")
    delete_parser.add_argument("--combined-model-path", default=None,
                               help="Path to a combined YOLO and Facenet .pt model.")

    # -------------------------------------------------------------------------
    # Export model
    # -------------------------------------------------------------------------
    export_model_parser = subparsers.add_parser("export-model", help="Export the current model and user data as a .pt file.")
    export_model_parser.add_argument("--export-path", required=True, help="Path to save the exported .pt model.")
    export_model_parser.add_argument("--model-path", default=None,
                                     help="Path or URL to custom YOLO model (.pt). If not set, default model is used.")
    export_model_parser.add_argument("--detector", default=None,
                                     help="Name of the detector plugin to use.")
    export_model_parser.add_argument("--embedder", default=None,
                                     help="Name of the embedder plugin to use.")
    export_model_parser.add_argument("--combined-model-path", default=None,
                                     help="Path to a combined YOLO and Facenet .pt model.")

    # -------------------------------------------------------------------------
    # Import model
    # -------------------------------------------------------------------------
    import_model_parser = subparsers.add_parser("import-model", help="Import a combined model from a .pt file.")
    import_model_parser.add_argument("--import-path", required=True, help="Path to the .pt model to import.")
    import_model_parser.add_argument("--model-path", default=None,
                                     help="Path or URL to custom YOLO model (.pt). If not set, default model is used.")
    import_model_parser.add_argument("--detector", default=None,
                                     help="Name of the detector plugin to use.")
    import_model_parser.add_argument("--embedder", default=None,
                                     help="Name of the embedder plugin to use.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Build the Config / FacialRecognition objects where needed
    # -------------------------------------------------------------------------
    # For commands that don't require FacialRecognition (e.g., export-model, import-model), handle separately.

    if args.command in ["register", "identify", "list-users", "delete-user"]:
        config = Config(
            conf_threshold=args.conf if hasattr(args, 'conf') else 0.5,
            yolo_model_path=args.model_path,
            detector_plugin=args.detector,
            embedder_plugin=args.embedder
        )
        fr = FacialRecognition(
            config,
            combined_model_path=args.combined_model_path
        )

    # -------------------------------------------------------------------------
    # Handle each command
    # -------------------------------------------------------------------------
    if args.command == "register":
        # Batch processing with parallelism
        def process_image(image_path):
            try:
                pil_image = Image.open(image_path).convert("RGB")
                return pil_image
            except Exception as e:
                print(f"[Error] Failed to open image {image_path}: {e}")
                return None

        pil_images = []
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            future_to_image = {executor.submit(process_image, img): img for img in args.images}
            for future in as_completed(future_to_image):
                img = future_to_image[future]
                result = future.result()
                if result:
                    pil_images.append(result)

        if not pil_images:
            print("[Error] No valid images to register.")
            sys.exit(1)

        msg = fr.register_user(args.user, pil_images)
        print(msg)

    elif args.command == "identify":
        try:
            pil_image = Image.open(args.image).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to open image {args.image}: {e}")
            sys.exit(1)

        results = fr.identify_user(pil_image, threshold=args.threshold)
        for i, r in enumerate(results):
            print(f"Face {i+1}: box={r['box']}, user_id='{r['user_id']}', similarity={r['similarity']:.2f}")

    elif args.command == "export-data":
        data_file = Config().user_data_path  # default user_faces.json
        if not os.path.exists(data_file):
            print("[Export] No user data found to export.")
            sys.exit(0)
        try:
            shutil.copyfile(data_file, args.output)
            print(f"[Export] Data exported to {args.output}")
        except Exception as e:
            print(f"[Export] Failed: {e}")

    elif args.command == "import-data":
        data_file = Config().user_data_path  # default user_faces.json
        if not os.path.exists(args.file):
            print(f"[Import] File '{args.file}' does not exist.")
            sys.exit(0)
        try:
            shutil.copyfile(args.file, data_file)
            print(f"[Import] Data imported from {args.file}")
        except Exception as e:
            print(f"[Import] Failed: {e}")

    elif args.command == "list-users":
        users = fr.list_users()
        if users:
            print("Registered Users:")
            for user in users:
                if hasattr(fr, 'combined_model'):
                    count = len(fr.combined_model.user_embeddings[user])
                else:
                    count = len(fr.user_data[user])
                print(f"- {user} ({count} embeddings)")
        else:
            print("No users registered.")

    elif args.command == "delete-user":
        success = fr.delete_user(args.user)
        if success:
            print(f"[Delete] User '{args.user}' has been deleted.")
        else:
            print(f"[Delete] User '{args.user}' does not exist.")

    elif args.command == "export-model":
        fr = FacialRecognition(
            Config(
                conf_threshold=0.5,
                yolo_model_path=args.model_path,
                detector_plugin=args.detector,
                embedder_plugin=args.embedder
            ),
            combined_model_path=args.combined_model_path
        )
        message = fr.export_model(args.export_path)
        print(message)

    elif args.command == "import-model":
        if not os.path.exists(args.import_path):
            print(f"[Import Model] File '{args.import_path}' does not exist.")
            sys.exit(1)
        fr = FacialRecognition.import_model(args.import_path, Config(
            conf_threshold=0.5,
            yolo_model_path=args.model_path,
            detector_plugin=args.detector,
            embedder_plugin=args.embedder
        ))
        print(f"[Import Model] Combined model loaded from {args.import_path}")

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
