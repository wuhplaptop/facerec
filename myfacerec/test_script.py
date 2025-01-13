"""
test_script.py

Demonstration of how to use facial_recognition.py without a YAML file.
"""

from facial_recognition import FacialRecognition
from PIL import Image

def main():
    # Example usage 1: Use the default downloaded model with a custom confidence threshold
    recognizer = FacialRecognition(
        yolo_model_path=None,        # None => automatically download face.pt from GitHub if needed
        user_data_path="user_faces.json",
        conf_threshold=0.75,         # <--- custom confidence threshold
        use_gpu=True
    )

    # Register a user (assuming we have face images locally)
    images = [
        Image.open("example_face1.jpg").convert("RGB"),
        Image.open("example_face2.jpg").convert("RGB"),
    ]
    register_msg = recognizer.register_user("Alice", images)
    print(register_msg)

    # Identify faces in a new image with a custom threshold for recognition
    test_image = Image.open("group_photo.jpg").convert("RGB")
    results = recognizer.identify_user(test_image, threshold=0.65)

    for i, r in enumerate(results):
        box = r['box']
        user_id = r['user_id']
        sim = r['similarity']
        print(f"Face {i+1}: box={box}, user_id='{user_id}', similarity={sim:.2f}")

    # Example usage 2: Provide your own YOLO model path instead
    # (commented out for demonstration)
    # custom_recognizer = FacialRecognition(
    #     yolo_model_path="my_custom_face_model.pt",
    #     conf_threshold=0.9
    # )
    # ... etc ...

if __name__ == "__main__":
    main()
