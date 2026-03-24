# ---------------------------------------------------------------------
# train_from_annotations.py
# FULL REWRITTEN PIPELINE (Frame extraction + Dataset + Training)
# Upgraded for:
# - Strong augmentation
# - Frozen ResNet18 early layers
# - LSTM dropout
# - AdamW optimizer
# - ReduceLROnPlateau scheduler
# - Proper train/val split BY VIDEO ID
# ---------------------------------------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_NO_CUDA"] = "1"
import math
import random
from tqdm import tqdm
from collections import Counter

import pandas as pd
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

# ================================================================
# USER CONFIG
# ================================================================
VIDEOS_DIR = r"F:\BMEapp\task1_actions\train"
FRAMES_DIR = r"F:\BMEapp\SaveFrame"
ANNOTATIONS_CSV = r"F:\BMEapp\task1_actions\annotations_train.csv"

CLIP_LEN = 32
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4 
NUM_WORKERS = 6

CHECKPOINT_DIR = "Newcheckpoints"
SKIP_EXTRACTION = True
FRAME_INDEX_BASE = 0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


# ================================================================
# Helper: Find a matching video file
# ================================================================
def find_video_file(video_id, videos_dir):
    files = [f for f in os.listdir(videos_dir) 
             if f.lower().endswith(('.mp4','.avi','.mov','.mkv','.wmv'))]
    for f in files:
        name = os.path.splitext(f)[0]
        if name == video_id or name.startswith(video_id):
            return os.path.join(videos_dir, f)
    return None


# ================================================================
# Extract frames for a single annotation row
# ================================================================
def extract_frames_for_row(video_path, clip_id, start_frame, stop_frame, out_root):
    out_folder = os.path.join(out_root, clip_id)
    os.makedirs(out_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    curr = start_frame
    idx = 0

    while curr <= stop_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_folder, f"{clip_id}_{idx}.jpg")
        cv2.imwrite(out_path, frame) 
        idx += 1
        curr += 1

    cap.release()


# ================================================================
# Extract all frames (once)
# ================================================================
def extract_all_from_annotations(df, videos_dir, out_root, skip_existing=True, frame_index_base=0):
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extract clips"):
        clip_id = str(row["clip_id"])
        video_id = str(row["video_id"])

        start_frame = int(row["start_frame"]) - frame_index_base
        stop_frame  = int(row["stop_frame"]) - frame_index_base

        video_path = find_video_file(video_id, videos_dir)
        if video_path is None:
            continue

        out_folder = os.path.join(out_root, clip_id)
        if skip_existing and os.path.isdir(out_folder) and len(os.listdir(out_folder)) > 0:
            continue

        extract_frames_for_row(video_path, clip_id, start_frame, stop_frame, out_root)


# ================================================================
# Dataset class
# ================================================================
class ClipActionDataset(Dataset):
    def __init__(self, csv_path, frames_root, clip_len=16, img_size=224, transform=None):
        self.df = pd.read_csv(csv_path)

        self.frames_root = frames_root
        self.clip_len = clip_len
        self.transform = transform

        labels = sorted(self.df["action"].unique())
        self.class2idx = {c:i for i,c in enumerate(labels)}
        self.idx2class = {i:c for c,i in self.class2idx.items()}

        self.samples = []
        for _, r in self.df.iterrows():
            clip_id = str(r["clip_id"])
            folder = os.path.join(self.frames_root, clip_id)

            if not os.path.isdir(folder):
                continue

            files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
            if not files:
                continue

            max_idx = max(int(f.split("_")[-1].split(".")[0]) for f in files)
            lbl = self.class2idx[r["action"]]
            

            self.samples.append((clip_id, 0, max_idx, lbl))

    def __len__(self):
        return len(self.samples)

    def _frame_path(self, clip_id, idx):
        return os.path.join(self.frames_root, clip_id, f"{clip_id}_{idx}.jpg")

    def _sample_indices(self, s, e):
        length = e - s + 1
        if length >= self.clip_len:
            interval = length / float(self.clip_len)
            return [int(s + i * interval) for i in range(self.clip_len)]
        else:
            idxs = list(range(s, e + 1))
            while len(idxs) < self.clip_len:
                idxs.append(idxs[-1])
            return idxs




    def __getitem__(self, idx):
        clip_id, s, e, label = self.samples[idx]
        idxs = self._sample_indices(s, e)

        imgs = []
        for i in idxs:
            img = Image.open(self._frame_path(clip_id, i)).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        frames = torch.stack(imgs, dim=0)
        return frames, label


# ================================================================
# Model: ResNet18 + LSTM (Frozen early layers, dropout added)
# ================================================================
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        
        
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        feature_dim = 512
        self.backbone.fc = nn.Identity()

        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if not ("layer3" in name or "layer4" in name):
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.backbone(x)          # (B*T, 512)
        feats = feats.view(B, T, -1)      # (B, T, 512)
        out, _ = self.lstm(feats)         # (B, T, 256)
        out = out[:, -1, :]               # Last frame output
        return self.classifier(out)


# ================================================================
# Training + Validation
# ================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0,0,0

    for frames, labels in tqdm(loader, desc="Train", leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss/total, correct/total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0,0,0

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Val", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * frames.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss/total, correct/total


# ================================================================
# MAIN
# ================================================================
def main():

    df = pd.read_csv(ANNOTATIONS_CSV)

    # Adjust index base
    if FRAME_INDEX_BASE == 1:
        df["start_frame"] -= 1
        df["stop_frame"]  -= 1

    # Extract frames
    if not SKIP_EXTRACTION:
        extract_all_from_annotations(df, VIDEOS_DIR, FRAMES_DIR)

    # Strong augmentation to prevent overfitting
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    dataset = ClipActionDataset(ANNOTATIONS_CSV, FRAMES_DIR, CLIP_LEN, IMG_SIZE, transform)

    # -------------------------
    # Train/val split BY VIDEO
    # -------------------------
    vids = dataset.df["video_id"].unique().tolist()
    random.shuffle(vids)

    split_point = int(0.8 * len(vids))
    train_videos = set(vids[:split_point])
    val_videos   = set(vids[split_point:])

    train_idx = [i for i,(clip,_,_,_) in enumerate(dataset.samples)
                 if dataset.df.iloc[i]["video_id"] in train_videos]

    val_idx   = [i for i,(clip,_,_,_) in enumerate(dataset.samples)
                 if dataset.df.iloc[i]["video_id"] in val_videos]

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    num_classes = len(dataset.class2idx)

    #model = ResNetLSTM(num_classes).to(device)
    model = ResNetLSTM(num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val = 0

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        # Scheduler reacts to validation loss
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("Saved new best model.")

    print("Training finished. Best Val Acc:", best_val)

    # -------------------------------
    # Plot metrics
    # -------------------------------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.show()

    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve")
    plt.show()


if __name__ == "__main__":
    main()
