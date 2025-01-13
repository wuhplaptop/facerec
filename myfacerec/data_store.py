# data_store.py

import abc
import json
import os
import numpy as np

class UserDataStore(abc.ABC):
    """
    Abstract interface for storing and retrieving user embeddings.
    """
    @abc.abstractmethod
    def load_user_data(self):
        pass

    @abc.abstractmethod
    def save_user_data(self, user_data):
        pass


class JSONUserDataStore(UserDataStore):
    def __init__(self, path="user_faces.json"):
        self.path = path
        # Load at init or on-demand if you prefer

    def load_user_data(self):
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                json.dump({}, f)
        with open(self.path, 'r') as f:
            data = json.load(f)
        # Convert lists to np.array
        for user_id in data:
            data[user_id] = [np.array(e) for e in data[user_id]]
        return data

    def save_user_data(self, user_data):
        serializable_data = {
            user_id: [emb.tolist() for emb in emb_list]
            for user_id, emb_list in user_data.items()
        }
        with open(self.path, 'w') as f:
            json.dump(serializable_data, f)
