import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import os
from myfacerec.config import Config
from myfacerec.facial_recognition import FacialRecognition

# Initialize configuration and FacialRecognition
DATA_PATH = "user_faces.json"  # Path to store user data

# Ensure the data path exists
if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'w') as f:
        f.write("{}")  # Initialize empty JSON

# Initialize rolo-rec
config = Config(
    conf_threshold=0.75,  # You can adjust this threshold
    user_data_path=DATA_PATH
)
fr = FacialRecognition(config)

# Streamlit App
st.set_page_config(page_title="Rolo-Rec Face Recognition App", layout="wide")
st.title("Rolo-Rec Face Recognition App")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Register User", "Identify Faces"]
choice = st.sidebar.radio("Go to", options)

if choice == "Register User":
    st.header("Register a New User")
    
    # Input for User ID
    user_id = st.text_input("Enter User ID", help="Unique identifier for the user (e.g., Alice)")
    
    # File uploader for images
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more images containing the user's face."
    )
    
    if st.button("Register User"):
        if not user_id:
            st.error("Please enter a User ID.")
        elif not uploaded_files:
            st.error("Please upload at least one image.")
        else:
            images = []
            failed_uploads = []
            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    images.append(image)
                except Exception as e:
                    failed_uploads.append(uploaded_file.name)
            
            if failed_uploads:
                st.error(f"Failed to upload the following files: {', '.join(failed_uploads)}")
            
            if images:
                with st.spinner("Registering user..."):
                    msg = fr.register_user(user_id, images)
                st.success(msg)
                st.balloons()

elif choice == "Identify Faces":
    st.header("Identify Faces in an Image")
    
    # File uploader for the image to identify
    identify_file = st.file_uploader(
        "Upload Image for Identification",
        type=["jpg", "jpeg", "png"],
        help="Upload an image in which to identify registered users."
    )
    
    if st.button("Identify"):
        if not identify_file:
            st.error("Please upload an image to identify.")
        else:
            try:
                image = Image.open(identify_file).convert("RGB")
                # Perform identification
                with st.spinner("Identifying faces..."):
                    results = fr.identify_user(image, threshold=0.65)  # Adjust similarity threshold as needed
                
                # Create a copy of the image to draw bounding boxes
                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)
                
                # Optionally, load a font
                try:
                    font = ImageFont.truetype("arial.ttf", size=15)
                except IOError:
                    font = ImageFont.load_default()
                
                for res in results:
                    box = res['box']  # (x1, y1, x2, y2)
                    user_id = res['user_id']
                    similarity = res['similarity']
                    # Draw rectangle
                    draw.rectangle(box, outline="red", width=2)
                    # Draw label
                    label = f"{user_id} ({similarity:.2f})" if user_id != "Unknown" else "Unknown"
                    
                    # Use draw.textbbox to calculate text size
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Draw a filled rectangle behind the text for better visibility
                    draw.rectangle([box[0], box[1] - text_height, box[0] + text_width, box[1]], fill="red")
                    draw.text((box[0], box[1] - text_height), label, fill="white", font=font)
                
                # Display the annotated image
                st.image(annotated_image, caption="Identification Results", use_column_width=True)
                
                if not results:
                    st.info("No faces detected.")
                else:
                    # Display results in a table
                    st.subheader("Detection Results")
                    for idx, res in enumerate(results, 1):
                        box = res['box']
                        user_id = res['user_id']
                        similarity = res['similarity']
                        st.write(f"**Face {idx}:**")
                        st.write(f"- **Bounding Box**: {box}")
                        st.write(f"- **User ID**: {user_id}")
                        st.write(f"- **Similarity**: {similarity:.2f}")
            except Exception as e:
                st.error(f"Error processing image: {e}")
