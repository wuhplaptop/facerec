# myfacerec/plugins/base.py

import importlib
import pkg_resources
import logging
from typing import Type

from ..detectors import FaceDetector
from ..embedders import FaceEmbedder
from ..data_store import UserDataStore

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Manages loading of detector, embedder, and data store plugins.
    Plugins must be registered via entry points in setup.py.
    """

    def load_detector(self, name: str) -> FaceDetector:
        try:
            entry_point = pkg_resources.get_entry_map('rolo-rec')['rolo_rec.detectors'][name]
            detector_class = entry_point.load()
            logger.info(f"Loaded detector plugin: {name}")
            return detector_class()
        except KeyError:
            logger.error(f"Detector plugin '{name}' not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to load detector plugin '{name}': {e}")
            raise

    def load_embedder(self, name: str) -> FaceEmbedder:
        try:
            entry_point = pkg_resources.get_entry_map('rolo-rec')['rolo_rec.embedders'][name]
            embedder_class = entry_point.load()
            logger.info(f"Loaded embedder plugin: {name}")
            return embedder_class()
        except KeyError:
            logger.error(f"Embedder plugin '{name}' not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedder plugin '{name}': {e}")
            raise

    def load_data_store(self, name: str) -> UserDataStore:
        try:
            entry_point = pkg_resources.get_entry_map('rolo-rec')['rolo_rec.data_stores'][name]
            data_store_class = entry_point.load()
            logger.info(f"Loaded data store plugin: {name}")
            return data_store_class()
        except KeyError:
            logger.error(f"Data store plugin '{name}' not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to load data store plugin '{name}': {e}")
            raise
