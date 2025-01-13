# cli.py

import argparse
import sys
from PIL import Image

from .config import Config
from .facial_recognition import FacialRecognition

def main():
    parser = argparse.ArgumentParser(prog="face-rec")
    subparsers = parser.add_subparsers(dest="command")

    # Register command
    reg_parser = subparsers.add_parser("register", help="Register a user.")
    reg_parser.add_argument("--user", required=True, help="User ID (unique).")
    reg_parser.add_argument("--images", nargs="+", required=True, help="Paths to one or more face images.")
    reg_parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")

    # Identify command
    id_parser = subparsers.add_parser("identify", help="Identify faces in an image.")
    id_parser.add_argument("--image", required=True, help="Path to input image.")
    id_parser.add_argument("--threshold", type=float, default=0.6, help="Recognition threshold.")
    id_parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = Config(conf_threshold=args.conf)
    fr = FacialRecognition(config)

    if args.command == "register":
        pil_images = [Image.open(p).convert("RGB") for p in args.images]
        msg = fr.register_user(args.user, pil_images)
        print(msg)

    elif args.command == "identify":
        pil_image = Image.open(args.image).convert("RGB")
        results = fr.identify_user(pil_image, threshold=args.threshold)
        for i, r in enumerate(results):
            print(f"Face {i+1}: box={r['box']}, user_id='{r['user_id']}', similarity={r['similarity']:.2f}")
