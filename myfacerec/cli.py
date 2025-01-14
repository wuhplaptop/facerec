import argparse
import sys
from PIL import Image
import json
import shutil
import os

from .config import Config
from .facial_recognition import FacialRecognition

def main():
    parser = argparse.ArgumentParser(prog="rolo-rec")
    subparsers = parser.add_subparsers(dest="command")

    # --------------------
    # Register command
    # --------------------
    reg_parser = subparsers.add_parser("register", help="Register a user.")
    reg_parser.add_argument("--user", required=True, help="User ID (unique).")
    reg_parser.add_argument("--images", nargs="+", required=True, help="Paths to one or more face images.")
    reg_parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    reg_parser.add_argument("--model-path", required=False, default=None,
                            help="Path to custom YOLO model (local file or URL). If not set, default model is used.")

    # --------------------
    # Identify command
    # --------------------
    id_parser = subparsers.add_parser("identify", help="Identify faces in an image.")
    id_parser.add_argument("--image", required=True, help="Path to input image.")
    id_parser.add_argument("--threshold", type=float, default=0.6, help="Recognition threshold.")
    id_parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    id_parser.add_argument("--model-path", required=False, default=None,
                          help="Path to custom YOLO model (local file or URL). If not set, default model is used.")

    # --------------------
    # Optional: Export data
    # --------------------
    export_parser = subparsers.add_parser("export-data", help="Export the face data (JSON) to a specified file.")
    export_parser.add_argument("--output", required=True, help="Path to save the exported data.")

    # --------------------
    # Optional: Import data
    # --------------------
    import_parser = subparsers.add_parser("import-data", help="Import face data from a JSON file.")
    import_parser.add_argument("--file", required=True, help="Path to the JSON file to import.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # For register/identify, we need the main FR object
    if args.command in ["register", "identify"]:
        config = Config(
            conf_threshold=args.conf,
            yolo_model_path=args.model_path  # Let the user override the model path
        )
        fr = FacialRecognition(config)

    # --------------------
    # Handle commands
    # --------------------
    if args.command == "register":
        pil_images = [Image.open(p).convert("RGB") for p in args.images]
        msg = fr.register_user(args.user, pil_images)
        print(msg)

    elif args.command == "identify":
        pil_image = Image.open(args.image).convert("RGB")
        results = fr.identify_user(pil_image, threshold=args.threshold)
        for i, r in enumerate(results):
            print(f"Face {i+1}: box={r['box']}, user_id='{r['user_id']}', similarity={r['similarity']:.2f}")

    elif args.command == "export-data":
        # Just copy or rename your user_data_path, or read & write it
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
