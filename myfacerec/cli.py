# cli.py
import argparse
import sys
from PIL import Image

from .config import Config
from .facial_recognition import FacialRecognition

def main():
    parser = argparse.ArgumentParser(prog="myfacerec")
    subparsers = parser.add_subparsers(dest="command")

    # Register sub-command
    register_parser = subparsers.add_parser("register", help="Register a user.")
    register_parser.add_argument("--user", required=True, help="User ID (unique).")
    register_parser.add_argument("--images", nargs="+", required=True, help="Paths to one or more face images.")
    register_parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for YOLO detection.")

    # Identify sub-command
    identify_parser = subparsers.add_parser("identify", help="Identify faces in an image.")
    identify_parser.add_argument("--image", required=True, help="Path to input image.")
    identify_parser.add_argument("--threshold", type=float, default=0.6, help="Face recognition threshold.")
    identify_parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for YOLO detection.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create config
    config = Config(conf_threshold=args.conf)
    fr = FacialRecognition(config)

    if args.command == "register":
        # Convert image paths to PIL images
        pil_images = [Image.open(p).convert("RGB") for p in args.images]
        msg = fr.register_user(args.user, pil_images)
        print(msg)

    elif args.command == "identify":
        pil_image = Image.open(args.image).convert("RGB")
        results = fr.identify_user(pil_image, threshold=args.threshold)
        for r in results:
            print(r)

if __name__ == "__main__":
    main()
