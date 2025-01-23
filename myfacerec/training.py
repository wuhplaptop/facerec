# myfacerec/training.py

import os

def train_yolo(data_dir, epochs, batch_size, save_path):
    """
    Example stub for YOLO training with ultralytics.
    The user must structure data_dir to have images/ and labels/ in YOLO format,
    plus a data.yaml file referencing those paths.
    """
    from ultralytics import YOLO

    # Suppose the user has a data.yaml in data_dir
    data_yaml = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"[Error] Could not find data.yaml in {data_dir}.")
        return

    # Start with a base YOLO model (like yolov8n.pt)
    model = YOLO("yolov8n.pt")  # or a custom path
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        name="face_training",  # folder name inside runs/
    )
    # Save best weights
    best_weights = os.path.join("runs", "detect", "face_training", "weights", "best.pt")
    if os.path.exists(best_weights):
        os.rename(best_weights, save_path)
        print(f"[Info] YOLO model training complete. Weights saved to {save_path}")
    else:
        print("[Warning] Best weights not found; training may have failed or path is incorrect.")

def train_facenet(data_dir, epochs, batch_size, save_path):
    """
    Example stub for FaceNet fine-tuning.
    This is a skeleton that you can fill with a real dataset class, transforms, etc.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from facenet_pytorch import InceptionResnetV1
    import numpy as np
    from PIL import Image

    class SimpleFaceDataset(Dataset):
        """
        Expects a folder structure: data_dir/user_id/*.jpg
        or data_dir/user_id/*.png
        Each user_id folder is a class label.
        """
        def __init__(self, root, transform=None):
            self.samples = []
            self.labels = []
            self.user_ids = []
            self.transform = transform

            user_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for idx, user in enumerate(user_dirs):
                user_path = os.path.join(root, user)
                for f in os.listdir(user_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(os.path.join(user_path, f))
                        self.labels.append(idx)
                self.user_ids.append(user)
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path = self.samples[idx]
            label = self.labels[idx]
            img = Image.open(path).convert("RGB")
            img = img.resize((160, 160))

            # Simple normalization: Facenet typically expects -1..1 
            img_np = np.array(img).astype(np.float32)/255.0
            img_np = (img_np - 0.5)/0.5
            # Turn into torch tensor
            img_tensor = torch.from_numpy(img_np).permute(2,0,1)  # (3,160,160)
            return img_tensor, label

    # Build dataset
    train_dataset = SimpleFaceDataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(set(train_dataset.labels))
    if num_classes < 2:
        print("[Error] At least 2 classes/users are needed to fine-tune FaceNet properly.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize FaceNet
    model = InceptionResnetV1(pretrained='vggface2')
    model.classify = True
    # Replace final layer to match num_classes
    model.last_linear = nn.Linear(512, num_classes)
    model.logits = nn.LogSoftmax(dim=-1)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), save_path)
    print(f"[Info] Fine-tuned FaceNet saved to {save_path}")
