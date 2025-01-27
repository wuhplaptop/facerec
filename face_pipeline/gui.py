# face_pipeline/gui.py

import os
import sys
import argparse
import math
import requests
import numpy as np
from PIL import Image
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import Counter
import logging
import pickle
import cv2
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import mediapipe as mp

from .config import PipelineConfig
from .core import FacePipeline
from .database import FaceDatabase
from .detectors import YOLOFaceDetector
from .trackers import FaceTracker
from .embedder import FaceNetEmbedder
from .utilities import (
    detect_blink,
    classify_eye_color,
    get_dominant_color,
    process_face_mesh,
    draw_face_mesh,
    detect_eye_color,
    HandTracker,
    save_pipeline,
    download_pipeline,
)

logger = logging.getLogger(__name__)

# Constants
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def rgb_to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
    return '#%02x%02x%02x' % rgb_tuple

def hex_to_rgb(hex_string: str) -> Tuple[int, int, int]:
    hex_string = hex_string.lstrip('#')
    return tuple(int(hex_string[i: i + 2], 16) for i in (0, 2, 4))

def suppress_stderr():
    """Suppress stderr by redirecting it to os.devnull."""
    import contextlib
    @contextlib.contextmanager
    def _suppress():
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    return _suppress()

def run_streamlit_gui():
    st.set_page_config(page_title="Face Recognition System", layout="wide")

    @st.cache_resource
    def load_pipeline_func():
        with suppress_stderr():
            config = PipelineConfig.load(config_path='face_pipeline/config.pkl')
            pipeline = FacePipeline(config)
            pipeline.initialize()
        return pipeline

    def load_pipeline_cleared():
        st.experimental_memo.clear()
        with suppress_stderr():
            config = PipelineConfig.load(config_path='face_pipeline/config.pkl')
            pipeline = FacePipeline(config)
            pipeline.initialize()
        return pipeline

    try:
        pipeline = load_pipeline_func()
        st.title("Real-Time Face Recognition System")

        with st.sidebar:
            st.header("Configuration")
            pipeline.config.recognition['enable'] = st.checkbox(
                "Enable Face Recognition", value=pipeline.config.recognition['enable'], key='enable_face_recognition'
            )
            pipeline.config.anti_spoof['enable'] = st.checkbox(
                "Enable Anti-Spoofing", value=pipeline.config.anti_spoof['enable'], key='enable_anti_spoofing'
            )
            pipeline.config.blink['enable'] = st.checkbox(
                "Enable Blink Detection", value=pipeline.config.blink['enable'], key='enable_blink_detection'
            )
            pipeline.config.hand['enable'] = st.checkbox(
                "Enable Hand Tracking", value=pipeline.config.hand['enable'], key='enable_hand_tracking'
            )
            pipeline.config.eye_color['enable'] = st.checkbox(
                "Enable Eye Color Detection", value=pipeline.config.eye_color['enable'], key='enable_eye_color_detection'
            )

            st.subheader("Face Mesh Options")
            pipeline.config.face_mesh_options['enable'] = st.checkbox(
                "Enable Face Mesh", value=pipeline.config.face_mesh_options['enable'], key='enable_face_mesh'
            )
            if pipeline.config.face_mesh_options['enable']:
                pipeline.config.face_mesh_options['tesselation'] = st.checkbox(
                    "Show Tesselation", value=pipeline.config.face_mesh_options['tesselation'], key='show_tesselation'
                )
                pipeline.config.face_mesh_options['contours'] = st.checkbox(
                    "Show Contours", value=pipeline.config.face_mesh_options['contours'], key='show_contours'
                )
                pipeline.config.face_mesh_options['irises'] = st.checkbox(
                    "Show Irises", value=pipeline.config.face_mesh_options['irises'], key='show_irises'
                )

            st.subheader("Confidence Thresholds")
            pipeline.config.detection_conf_thres = st.slider(
                "Detection Confidence Threshold", 0.0, 1.0, pipeline.config.detection_conf_thres, key='detection_conf_threshold'
            )
            pipeline.config.recognition_conf_thres = st.slider(
                "Recognition Similarity Threshold", 0.5, 1.0, pipeline.config.recognition_conf_thres, key='recognition_similarity_threshold'
            )

            pipeline.config.anti_spoof['lap_thresh'] = st.slider(
                "Anti-Spoof Threshold", 0.0, 200.0, pipeline.config.anti_spoof['lap_thresh'], key='anti_spoof_threshold'
            )
            pipeline.config.blink['ear_thresh'] = st.slider(
                "Blink EAR Threshold", 0.0, 0.5, pipeline.config.blink['ear_thresh'], key='blink_ear_threshold'
            )
            pipeline.config.hand['min_detection_confidence'] = st.slider(
                "Hand Detection Confidence", 0.0, 1.0, pipeline.config.hand['min_detection_confidence'], key='hand_detection_confidence'
            )
            pipeline.config.hand['min_tracking_confidence'] = st.slider(
                "Hand Tracking Confidence", 0.0, 1.0, pipeline.config.hand['min_tracking_confidence'], key='hand_tracking_confidence'
            )

            st.header("Color Options")

            st.subheader("Bounding Boxes")
            pipeline.config.bbox_color = hex_to_rgb(
                st.color_picker("Box Color (Recognized)", rgb_to_hex(pipeline.config.bbox_color), key='bbox_color_recognized')
            )
            pipeline.config.spoofed_bbox_color = hex_to_rgb(
                st.color_picker("Box Color (Spoofed)", rgb_to_hex(pipeline.config.spoofed_bbox_color), key='bbox_color_spoofed')
            )
            pipeline.config.unknown_bbox_color = hex_to_rgb(
                st.color_picker("Box Color (Unknown)", rgb_to_hex(pipeline.config.unknown_bbox_color), key='bbox_color_unknown')
            )

            st.subheader("Blink Detection")
            pipeline.config.eye_outline_color = hex_to_rgb(
                st.color_picker("Eye Outline Color", rgb_to_hex(pipeline.config.eye_outline_color), key='eye_outline_color')
            )
            pipeline.config.blink_text_color = hex_to_rgb(
                st.color_picker("Blink Text Color", rgb_to_hex(pipeline.config.blink_text_color), key='blink_text_color')
            )

            st.subheader("Hand Tracking")
            pipeline.config.hand_landmark_color = hex_to_rgb(
                st.color_picker("Hand Landmark Color", rgb_to_hex(pipeline.config.hand_landmark_color), key='hand_landmark_color')
            )
            pipeline.config.hand_connection_color = hex_to_rgb(
                st.color_picker("Hand Connection Color", rgb_to_hex(pipeline.config.hand_connection_color), key='hand_connection_color')
            )
            pipeline.config.hand_text_color = hex_to_rgb(
                st.color_picker("Hand Text Color", rgb_to_hex(pipeline.config.hand_text_color), key='hand_text_color')
            )

            st.subheader("Face Mesh Colors")
            mesh_hex_color = st.color_picker("Mesh Color", rgb_to_hex(pipeline.config.mesh_color), key='mesh_color')
            pipeline.config.mesh_color = hex_to_rgb(mesh_hex_color)
            pipeline.config.contour_color = hex_to_rgb(
                st.color_picker("Contour Color", rgb_to_hex(pipeline.config.contour_color), key='contour_color')
            )
            pipeline.config.iris_color = hex_to_rgb(
                st.color_picker("Iris Color", rgb_to_hex(pipeline.config.iris_color), key='iris_color')
            )

            st.subheader("Eye Color Detection")
            pipeline.config.eye_color_text_color = hex_to_rgb(
                st.color_picker("Eye Color Text Color", rgb_to_hex(pipeline.config.eye_color_text_color), key='eye_color_text_color')
            )

            if st.button("Save Configuration"):
                pipeline.config.save('face_pipeline/config.pkl')
                st.success("Configuration saved!")

            st.header("Database Management")
            with st.expander("User Enrollment", expanded=False):
                enroll_name = st.text_input("Enter Name for Enrollment", key='enroll_name_input')
                uploaded_files = st.file_uploader(
                    "Upload Enrollment Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key='enroll_images'
                )
                if st.button("Enroll User") and enroll_name and uploaded_files:
                    with st.spinner("Enrolling user..."):
                        try:
                            for uploaded_file in uploaded_files:
                                image = cv2.imdecode(
                                    np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
                                )
                                if image is None:
                                    continue
                                detections = pipeline.detector.detect(
                                    image, pipeline.config.detection_conf_thres
                                )
                                if not detections:
                                    continue
                                for x1, y1, x2, y2, conf, cls in detections:
                                    face_roi = image[y1:y2, x1:x2]
                                    if face_roi.size == 0:
                                        continue
                                    emb = pipeline.facenet.get_embedding(face_roi)
                                    if emb is not None:
                                        pipeline.db.add_embedding(enroll_name, emb)
                            pipeline.db.save()
                            st.success(f"Enrolled {enroll_name} successfully!")
                        except Exception as e:
                            st.error(f"Enrollment failed: {str(e)}")

            with st.expander("User Search", expanded=False):
                search_mode = st.radio("Search Database By:", ["Name", "Image"], key='search_mode_radio')
                if search_mode == "Name":
                    search_name = st.selectbox(
                        "Select User to Search", options=[""] + pipeline.db.list_labels(), key='search_name_select'
                    )
                    if st.button("Search by Name") and search_name:
                        embeddings = pipeline.db.get_embeddings_by_label(search_name)
                        if embeddings:
                            st.write(f"Embeddings for user: {search_name}")
                            st.write(f"Number of embeddings: {len(embeddings)}")
                            st.write(
                                "First embedding (first 5 values):",
                                embeddings[0][:5] if embeddings else "No embeddings",
                            )
                        else:
                            st.warning(f"No embeddings found for user: {search_name}")
                elif search_mode == "Image":
                    search_image_file = st.file_uploader(
                        "Upload Image to Search", type=["jpg", "png", "jpeg"], key='search_image_upload'
                    )
                    if st.button("Search by Image") and search_image_file:
                        try:
                            search_image = cv2.imdecode(
                                np.frombuffer(search_image_file.read(), np.uint8), cv2.IMREAD_COLOR
                            )
                            if search_image is not None:
                                detections = pipeline.detector.detect(
                                    search_image, pipeline.config.detection_conf_thres
                                )
                                if detections:
                                    face_roi = search_image[
                                        detections[0][1]:detections[0][3],
                                        detections[0][0]:detections[0][2],
                                    ]
                                    if face_roi.size > 0:
                                        query_embedding = pipeline.facenet.get_embedding(face_roi)
                                        if query_embedding is not None:
                                            search_results = pipeline.db.search_by_image(
                                                query_embedding, pipeline.config.recognition_conf_thres
                                            )
                                            if search_results:
                                                st.subheader("Search Results:")
                                                for label, similarity in search_results:
                                                    st.write(f"- Name: {label}, Similarity: {similarity:.4f}")
                                            else:
                                                st.info(
                                                    "No similar faces found in the database with current threshold."
                                                )
                                        else:
                                            st.error("Could not generate embedding from uploaded image face.")
                                    else:
                                        st.warning("No valid face ROI found in the uploaded image.")
                                else:
                                    st.warning("No faces detected in the uploaded image.")
                            else:
                                st.error("Failed to decode uploaded image.")
                        except Exception as e:
                            st.error(f"Image search failed: {str(e)}")

        with st.expander("User Management Tools", expanded=False):
            if st.button("List Enrolled Users"):
                st.write("Enrolled Users:", pipeline.db.list_labels())
            remove_user = st.selectbox(
                "Select User to Remove", options=[""] + pipeline.db.list_labels(), key='remove_user_select'
            )
            if st.button("Remove Selected User") and remove_user:
                pipeline.db.remove_label(remove_user)
                pipeline.db.save()
                st.success(f"Removed user: {remove_user}")
            if st.button("Clear Cache"):
                st.experimental_memo.clear()
                st.success("Cache cleared!")

        tabs = st.tabs(["Real-Time Recognition", "Image Test"])
        with tabs[0]:
            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                try:
                    img = frame.to_ndarray(format="bgr24")
                    processed_img, detections = pipeline.process_frame(img)
                    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
                except Exception as e:
                    logger.error(f"Video processing error: {str(e)}")
                    return frame

            webrtc_ctx = webrtc_streamer(
                key="face-recognition",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with tabs[1]:
            st.header("Image Test")
            test_image = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"], key='image_test_upload_2')
            if test_image:
                try:
                    image = Image.open(test_image).convert('RGB')
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    processed_image, detections = pipeline.process_frame(image_bgr)
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_container_width=True)
                    with col2:
                        st.subheader("Processed Image")
                        st.image(processed_rgb, use_container_width=True)

                    if detections:
                        st.subheader("Detection Details")
                        for det in detections:
                            detection_details = {
                                "Track ID": det['track_id'],
                                "Name": det['name'],
                                "Confidence": det['confidence'],
                                "Bounding Box": det['bbox'],
                                "Blink Detected": det['blink'],
                                "Face Mesh": det['face_mesh'],
                                "Hands Detected": det['hands_detected'],
                                "Hand Count": det['hand_count'],
                                "Eye Color": det['eye_color'],
                            }
                            st.json(detection_details)
                    else:
                        st.warning("No faces detected in the image")

                except Exception as e:
                    st.error(f"Image processing failed: {str(e)}")
                    logger.exception(e)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.critical(f"Streamlit GUI failed: {str(e)}")
        logger.exception(e)
