# ================================================================
# TWO MODEL PIPELINE (Majority + Minority)
# ResNet18 + LSTM Action Recognition
# With Loss/Accuracy Curves + Confusion Matrix
# ================================================================

import os
import random
from tqdm import tqdm
from collections import Counter

import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ================================================================
# USER CONFIG
# ================================================================

FRAMES_DIR      = r"F:\BMEapp\SaveFrame"
ANNOTATIONS_CSV = r"F:\BMEapp\task1_actions\annotations_train.csv"

CLIP_LEN    = 24
IMG_SIZE    = 224
BATCH_SIZE  = 8
EPOCHS      = 40
LR          = 1e-4
NUM_WORKERS = 4
PATIENCE    = 6
THRESHOLD   = 15

CHECKPOINT_DIR = "TwoModelCheckpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ================================================================
# Dataset
# ================================================================

class ClipActionDataset(Dataset):

    def __init__(self, csv_path, frames_root, clip_len=16, transform=None):

        self.df          = pd.read_csv(csv_path)
        self.frames_root = frames_root
        self.clip_len    = clip_len
        self.transform   = transform

        labels           = sorted(self.df["action"].unique())
        self.class2idx   = {c: i for i, c in enumerate(labels)}
        self.idx2class   = {i: c for c, i in self.class2idx.items()}

        self.samples = []

        for _, r in self.df.iterrows():

            clip_id = str(r["clip_id"])
            folder  = os.path.join(self.frames_root, clip_id)

            if not os.path.isdir(folder):
                continue

            files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
            if not files:
                continue

            max_idx = max(int(f.split("_")[-1].split(".")[0]) for f in files)
            lbl     = self.class2idx[r["action"]]

            self.samples.append((clip_id, 0, max_idx, lbl, r["video_id"]))

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
        clip_id, s, e, label, _ = self.samples[idx]
        idxs = self._sample_indices(s, e)
        imgs = []
        for i in idxs:
            img = Image.open(self._frame_path(clip_id, i)).convert("RGB")
            img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs, dim=0), label


# ================================================================
# Model
# ================================================================

class ResNetLSTM(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.backbone    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) #models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        for name, param in self.backbone.named_parameters():
            if not ("layer3" in name or "layer4" in name):
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256, # Reduced hidden size to 128 to prevent overfitting on minority classes
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x     = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out    = out.mean(dim=1)
        return self.classifier(out)


# ================================================================
# Train / Validate
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss, correct, total = 0, 0, 0

    for frames, labels in tqdm(loader, leave=False):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * frames.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):

    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * frames.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def get_predictions(model, loader, device):

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames, labels in loader:
            frames  = frames.to(device)
            outputs = model(frames)
            preds   = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ================================================================
# Remapped Subset
# ================================================================

class RemappedSubset(Dataset):

    def __init__(self, dataset, indices, label_map):
        self.dataset   = dataset
        self.indices   = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx      = self.indices[idx]
        frames, label = self.dataset[real_idx]
        return frames, self.label_map[label]


# ================================================================
# Plotting helpers
# ================================================================

def plot_curves(history, model_name):
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax1.plot(epochs, history["val_acc"],   label="Val Acc",   marker="o", markersize=3)
    ax1.set_title(f"{model_name} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax2.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    ax2.set_title(f"{model_name} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(CHECKPOINT_DIR, f"curves_{model_name.lower()}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(CHECKPOINT_DIR, f"cm_{title.split()[3].lower()}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# ================================================================
# MAIN
# ================================================================

def main():

    device = "cpu"

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ClipActionDataset(ANNOTATIONS_CSV, FRAMES_DIR, CLIP_LEN, train_transform)
    val_dataset   = ClipActionDataset(ANNOTATIONS_CSV, FRAMES_DIR, CLIP_LEN, val_transform)

    # ================= CLASS SPLIT =================

    labels_list  = [s[3] for s in train_dataset.samples]
    class_counts = Counter(labels_list)

    majority_classes = [cls for cls, cnt in class_counts.items() if cnt >= THRESHOLD]
    minority_classes = [cls for cls, cnt in class_counts.items() if cnt <  THRESHOLD]

    print("Majority:", majority_classes)
    print("Minority:", minority_classes)

    major_map = {cls: i for i, cls in enumerate(majority_classes)}
    minor_map = {cls: i for i, cls in enumerate(minority_classes)}

    major_idx2cls = {i: cls for cls, i in major_map.items()}
    minor_idx2cls = {i: cls for cls, i in minor_map.items()}

    idx2name = train_dataset.idx2class

    # ================= TRAIN VAL SPLIT =================

    videos    = list(set([s[4] for s in train_dataset.samples]))
    random.shuffle(videos)
    split     = int(0.8 * len(videos))
    train_vids = set(videos[:split])
    val_vids   = set(videos[split:])

    train_idx = [i for i, s in enumerate(train_dataset.samples) if s[4] in train_vids]
    val_idx   = [i for i, s in enumerate(val_dataset.samples)   if s[4] in val_vids]

    def filter_indices(indices, samples, allowed_classes):
        return [i for i in indices if samples[i][3] in allowed_classes]

    train_major_idx = filter_indices(train_idx, train_dataset.samples, majority_classes)
    train_minor_idx = filter_indices(train_idx, train_dataset.samples, minority_classes)
    val_major_idx   = filter_indices(val_idx,   val_dataset.samples,   majority_classes)
    val_minor_idx   = filter_indices(val_idx,   val_dataset.samples,   minority_classes)

    train_major = RemappedSubset(train_dataset, train_major_idx, major_map)
    train_minor = RemappedSubset(train_dataset, train_minor_idx, minor_map)
    val_major   = RemappedSubset(val_dataset,   val_major_idx,   major_map)
    val_minor   = RemappedSubset(val_dataset,   val_minor_idx,   minor_map)

    train_major_loader = DataLoader(train_major, BATCH_SIZE, True,  num_workers=NUM_WORKERS)
    train_minor_loader = DataLoader(train_minor, BATCH_SIZE, True,  num_workers=NUM_WORKERS)
    val_major_loader   = DataLoader(val_major,   BATCH_SIZE, False, num_workers=NUM_WORKERS)
    val_minor_loader   = DataLoader(val_minor,   BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # ================= MODELS =================

    model_major = ResNetLSTM(len(majority_classes)).to(device)
    model_minor = ResNetLSTM(len(minority_classes)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer_major = torch.optim.AdamW(model_major.parameters(), lr=LR)
    optimizer_minor = torch.optim.AdamW(model_minor.parameters(), lr=LR)

    scheduler_major = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_major, T_max=EPOCHS)
    scheduler_minor = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_minor, T_max=EPOCHS)

    # ================= HISTORY =================

    history_major = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    history_minor = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_major       = 0
    best_minor       = 0
    no_improve_major = 0
    no_improve_minor = 0
    major_done       = False
    minor_done       = False

    # ================= TRAIN LOOP =================

    for epoch in range(1, EPOCHS + 1):

        print(f"\n===== Epoch {epoch} =====")

        if not major_done:

            train_loss, train_acc = train_one_epoch(
                model_major, train_major_loader, criterion, optimizer_major, device
            )
            val_loss, val_acc = validate(
                model_major, val_major_loader, criterion, device
            )
            scheduler_major.step()

            history_major["train_loss"].append(train_loss)
            history_major["val_loss"].append(val_loss)
            history_major["train_acc"].append(train_acc)
            history_major["val_acc"].append(val_acc)

            print(f"[MAJOR] Train {train_acc:.5f} | Val {val_acc:.5f}")

            if val_acc > best_major:
                best_major       = val_acc
                no_improve_major = 0
                torch.save(model_major.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "major_best.pth"))
                print(f"[MAJOR] Saved best ({best_major:.5f})")
            else:
                no_improve_major += 1
                if no_improve_major >= PATIENCE:
                    print("[MAJOR] Early stopping")
                    major_done = True

        if not minor_done:

            train_loss, train_acc = train_one_epoch(
                model_minor, train_minor_loader, criterion, optimizer_minor, device
            )
            val_loss, val_acc = validate(
                model_minor, val_minor_loader, criterion, device
            )
            scheduler_minor.step()

            history_minor["train_loss"].append(train_loss)
            history_minor["val_loss"].append(val_loss)
            history_minor["train_acc"].append(train_acc)
            history_minor["val_acc"].append(val_acc)

            print(f"[MINOR] Train {train_acc:.5f} | Val {val_acc:.5f}")

            if val_acc > best_minor:
                best_minor       = val_acc
                no_improve_minor = 0
                torch.save(model_minor.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "minor_best.pth"))
                print(f"[MINOR] Saved best ({best_minor:.5f})")
            else:
                no_improve_minor += 1
                if no_improve_minor >= PATIENCE:
                    print("[MINOR] Early stopping")
                    minor_done = True

        if major_done and minor_done:
            print("\nBoth models early stopped.")
            break

    print("\nTraining Finished")
    print(f"Best Major Val Acc: {best_major:.5f}")
    print(f"Best Minor Val Acc: {best_minor:.5f}")

    # ================= CURVES =================

    plot_curves(history_major, "MAJOR")
    plot_curves(history_minor, "MINOR")

    # ================= CONFUSION MATRIX =================

    model_major.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "major_best.pth"), map_location=device)
    )
    model_minor.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "minor_best.pth"), map_location=device)
    )

    print("\nEvaluating MAJOR model...")
    true_major, pred_major = get_predictions(model_major, val_major_loader, device)

    print("Evaluating MINOR model...")
    true_minor, pred_minor = get_predictions(model_minor, val_minor_loader, device)

    major_class_names = [str(idx2name[major_idx2cls[i]]) for i in range(len(majority_classes))]
    minor_class_names = [str(idx2name[minor_idx2cls[i]]) for i in range(len(minority_classes))]

    # MAJOR — all classes present so no filtering needed
    cm_major = confusion_matrix(true_major, pred_major)
    plot_confusion_matrix(cm_major, major_class_names, "Confusion Matrix — MAJOR Model")

    print("\n===== MAJOR Model Report =====")
    print(classification_report(true_major, pred_major, target_names=major_class_names))

    # MINOR — filter to only classes that appear in val set
    present_labels = sorted(np.unique(np.concatenate([true_minor, pred_minor])))
    minor_class_names_present = [str(idx2name[minor_idx2cls[i]]) for i in present_labels]

    cm_minor = confusion_matrix(true_minor, pred_minor, labels=present_labels)
    plot_confusion_matrix(cm_minor, minor_class_names_present, "Confusion Matrix — MINOR Model")

    print("\n===== MINOR Model Report =====")
    print(classification_report(true_minor, pred_minor, labels=present_labels, target_names=minor_class_names_present))


if __name__ == "__main__":
    main()