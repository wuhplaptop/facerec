# face_pipeline/utilities.py

import requests
import os
import logging

logger = logging.getLogger(__name__)

def upload_pipeline(pipeline_path: str, destination_url: str) -> bool:
    try:
        with open(pipeline_path, 'rb') as f:
            response = requests.post(destination_url, files={'file': f})
        if response.status_code == 200:
            logger.info(f"Successfully uploaded pipeline to {destination_url}")
            return True
        else:
            logger.error(f"Failed to upload pipeline. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Upload pipeline failed: {str(e)}")
        return False

def download_pipeline(file_url: str, save_path: str) -> bool:
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded pipeline from {file_url} to {save_path}")
            return True
        else:
            logger.error(f"Failed to download pipeline. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Download pipeline failed: {str(e)}")
        return False

def save_pipeline(pipeline: 'FacePipeline', save_path: str):
    try:
        pipeline.config.save(save_path)
        # Assuming the model is part of the pipeline's config
        # Add any additional files if necessary
        logger.info(f"Saved pipeline configuration to {save_path}")
    except Exception as e:
        logger.error(f"Save pipeline failed: {str(e)}")
        raise
