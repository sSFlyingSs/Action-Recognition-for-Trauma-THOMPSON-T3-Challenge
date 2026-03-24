# ================================================================
# TWO-TIER PIPELINE
# Tier 1: MAJOR ResNet18+LSTM Classifier (count >= 20)
# Tier 2: Few-Shot Embedding Matcher     (count <  20)
# ================================================================

import os
import random
from tqdm import tqdm
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
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

CLIP_LEN        = 24
IMG_SIZE        = 224
BATCH_SIZE      = 8
EPOCHS          = 50
LR              = 1e-4
NUM_WORKERS     = 4
PATIENCE        = 5
THRESHOLD       = 20        # MAJOR if count >= 20, else few-shot
RANDOM_SEED     = 42

CHECKPOINT_DIR  = "TwoTierCheckpoints"
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

        labels         = sorted(self.df["action"].unique())
        self.class2idx = {c: i for i, c in enumerate(labels)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}

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
# Models
# ================================================================

class ResNetLSTM(nn.Module):
    """Standard classifier for MAJOR classes."""

    def __init__(self, num_classes):
        super().__init__()

        self.backbone    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        for name, param in self.backbone.named_parameters():
            if not ("layer3" in name or "layer4" in name):
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
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
        B, T, C, H, W = x.shape
        x      = x.view(B * T, C, H, W)
        feats  = self.backbone(x)
        feats  = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out    = out.mean(dim=1)
        return self.classifier(out)

    def get_logits_and_probs(self, x):
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=1)
        return logits, probs


class ResNetLSTMEmbedder(nn.Module):
    """Embedding backbone for few-shot matching — no classifier head."""

    def __init__(self):
        super().__init__()

        self.backbone    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        for name, param in self.backbone.named_parameters():
            if not ("layer3" in name or "layer4" in name):
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )

        # Extra projection layer for better embedding space
        self.proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x      = x.view(B * T, C, H, W)
        feats  = self.backbone(x)
        feats  = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        out    = out.mean(dim=1)
        emb    = self.proj(out)
        return F.normalize(emb, dim=1)  # L2 normalize for cosine similarity


# ================================================================
# Few-Shot Embedding Store
# ================================================================

class FewShotMatcher:
    """
    Stores embeddings per class and matches via cosine similarity.
    Uses mean prototype per class (prototypical network style).
    """

    def __init__(self):
        self.prototypes = {}   # class_id -> mean embedding tensor

    def build(self, embedder, loader, idx2class, device):
        """Compute and store mean prototype for each class."""

        embedder.eval()
        class_embeddings = defaultdict(list)

        print("Building few-shot prototypes...")

        with torch.no_grad():
            for frames, labels in tqdm(loader, leave=False):
                frames = frames.to(device)
                embs   = embedder(frames)
                for emb, lbl in zip(embs, labels):
                    class_embeddings[lbl.item()].append(emb.cpu())

        for cls_id, embs in class_embeddings.items():
            stacked          = torch.stack(embs, dim=0)
            prototype        = stacked.mean(dim=0)
            prototype        = F.normalize(prototype, dim=0)
            self.prototypes[cls_id] = prototype

        print(f"Built prototypes for {len(self.prototypes)} classes.")

    def predict(self, embedding):
        """Return best matching class id and similarity score."""

        best_cls  = None
        best_sim  = -1.0

        for cls_id, proto in self.prototypes.items():
            sim = F.cosine_similarity(
                embedding.unsqueeze(0),
                proto.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_cls = cls_id

        return best_cls, best_sim

    def save(self, path):
        torch.save(self.prototypes, path)
        print(f"Saved prototypes: {path}")

    def load(self, path):
        self.prototypes = torch.load(path)
        print(f"Loaded prototypes: {path}")


# ================================================================
# Train / Validate (MAJOR)
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


# ================================================================
# Train Embedder (Few-Shot) with Triplet Loss
# ================================================================

def train_embedder_one_epoch(embedder, loader, optimizer, device):
    """
    Train the embedder using online triplet loss.
    Anchor and positive are same class, negative is different class.
    """

    embedder.train()
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a, b: 1 - F.cosine_similarity(a, b),
        margin=0.3
    )

    total_loss = 0
    count      = 0

    # Collect all embeddings and labels in batch
    all_embs   = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            embs   = embedder(frames)
            all_embs.append(embs.cpu())
            all_labels.append(labels)

    all_embs   = torch.cat(all_embs,   dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Online triplet mining
    optimizer.zero_grad()
    loss = torch.tensor(0.0, requires_grad=True)

    label_to_indices = defaultdict(list)
    for i, lbl in enumerate(all_labels.tolist()):
        label_to_indices[lbl].append(i)

    triplets_found = 0

    for lbl, indices in label_to_indices.items():
        if len(indices) < 2:
            continue

        neg_indices = [i for i in range(len(all_labels)) if all_labels[i] != lbl]
        if not neg_indices:
            continue

        for a_idx in indices:
            p_idx = random.choice([i for i in indices if i != a_idx])
            n_idx = random.choice(neg_indices)

            anchor   = all_embs[a_idx].to(device).unsqueeze(0)
            positive = all_embs[p_idx].to(device).unsqueeze(0)
            negative = all_embs[n_idx].to(device).unsqueeze(0)

            loss = loss + triplet_loss_fn(anchor, positive, negative)
            triplets_found += 1

    if triplets_found > 0:
        loss = loss / triplets_found
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss = loss.item()

    return total_loss


# ================================================================
# Plotting
# ================================================================

def plot_curves(history, model_name):

    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax1.plot(epochs, history["val_acc"],   label="Val Acc",   marker="o", markersize=3)
    ax1.set_title(f"{model_name} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

    tag       = "major" if "MAJOR" in title else "fewshot"
    save_path = os.path.join(CHECKPOINT_DIR, f"cm_{tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# ================================================================
# MAIN
# ================================================================

def main():

    random.seed(RANDOM_SEED)
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
    fewshot_classes  = [cls for cls, cnt in class_counts.items() if cnt <  THRESHOLD]

    print(f"MAJOR classes  : {len(majority_classes)}")
    print(f"Few-shot classes: {len(fewshot_classes)}")

    major_map   = {cls: i for i, cls in enumerate(majority_classes)}
    fewshot_map = {cls: i for i, cls in enumerate(fewshot_classes)}

    major_idx2cls   = {i: cls for cls, i in major_map.items()}
    fewshot_idx2cls = {i: cls for cls, i in fewshot_map.items()}

    idx2name = train_dataset.idx2class

    # ================= TRAIN VAL SPLIT =================

    videos     = list(set([s[4] for s in train_dataset.samples]))
    random.shuffle(videos)
    split      = int(0.8 * len(videos))
    train_vids = set(videos[:split])
    val_vids   = set(videos[split:])

    train_idx = [i for i, s in enumerate(train_dataset.samples) if s[4] in train_vids]
    val_idx   = [i for i, s in enumerate(val_dataset.samples)   if s[4] in val_vids]

    def filter_indices(indices, samples, allowed_classes):
        return [i for i in indices if samples[i][3] in allowed_classes]

    train_major_idx   = filter_indices(train_idx, train_dataset.samples, majority_classes)
    train_fewshot_idx = filter_indices(train_idx, train_dataset.samples, fewshot_classes)
    val_major_idx     = filter_indices(val_idx,   val_dataset.samples,   majority_classes)
    val_fewshot_idx   = filter_indices(val_idx,   val_dataset.samples,   fewshot_classes)

    train_major   = RemappedSubset(train_dataset, train_major_idx,   major_map)
    train_fewshot = RemappedSubset(train_dataset, train_fewshot_idx, fewshot_map)
    val_major     = RemappedSubset(val_dataset,   val_major_idx,     major_map)
    val_fewshot   = RemappedSubset(val_dataset,   val_fewshot_idx,   fewshot_map)

    train_major_loader   = DataLoader(train_major,   BATCH_SIZE, True,  num_workers=NUM_WORKERS)
    train_fewshot_loader = DataLoader(train_fewshot, BATCH_SIZE, True,  num_workers=NUM_WORKERS)
    val_major_loader     = DataLoader(val_major,     BATCH_SIZE, False, num_workers=NUM_WORKERS)
    val_fewshot_loader   = DataLoader(val_fewshot,   BATCH_SIZE, False, num_workers=NUM_WORKERS)

    # ================= MAJOR MODEL =================

    model_major     = ResNetLSTM(len(majority_classes)).to(device)
    criterion       = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_major = torch.optim.AdamW(model_major.parameters(), lr=LR)
    scheduler_major = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_major, T_max=EPOCHS)

    history_major  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_major     = 0
    no_improve     = 0
    major_done     = False

    print("\n========== Training MAJOR Model ==========")

    for epoch in range(1, EPOCHS + 1):

        if major_done:
            break

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

        print(f"Epoch {epoch:>3} | Train {train_acc:.5f} | Val {val_acc:.5f}")

        if val_acc > best_major:
            best_major = val_acc
            no_improve = 0
            torch.save(model_major.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "major_best.pth"))
            print(f"  Saved best ({best_major:.5f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("  Early stopping.")
                major_done = True

    print(f"\nBest MAJOR Val Acc: {best_major:.5f}")
    plot_curves(history_major, "MAJOR")

    # ================= FEW-SHOT EMBEDDER =================

    embedder           = ResNetLSTMEmbedder().to(device)
    optimizer_embedder = torch.optim.AdamW(embedder.parameters(), lr=LR)
    scheduler_embedder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_embedder, T_max=EPOCHS)

    EMBEDDER_EPOCHS = 20   # fewer epochs needed for triplet training
    best_emb_loss   = float("inf")

    print("\n========== Training Few-Shot Embedder ==========")

    for epoch in range(1, EMBEDDER_EPOCHS + 1):

        loss = train_embedder_one_epoch(
            embedder, train_fewshot_loader, optimizer_embedder, device
        )
        scheduler_embedder.step()

        print(f"Epoch {epoch:>3} | Triplet Loss {loss:.5f}")

        if loss < best_emb_loss:
            best_emb_loss = loss
            torch.save(embedder.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "embedder_best.pth"))
            print(f"  Saved best embedder ({best_emb_loss:.5f})")

    # Load best embedder and build prototypes from ALL train few-shot data
    embedder.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "embedder_best.pth"), map_location=device)
    )

    matcher = FewShotMatcher()
    matcher.build(embedder, train_fewshot_loader, idx2name, device)
    matcher.save(os.path.join(CHECKPOINT_DIR, "fewshot_prototypes.pth"))

    # ================= EVALUATION =================

    print("\n========== Evaluation ==========")

    # Load best major model
    model_major.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "major_best.pth"), map_location=device)
    )
    model_major.eval()
    embedder.eval()

    # --- MAJOR confusion matrix ---
    all_true_major, all_pred_major = [], []

    with torch.no_grad():
        for frames, labels in val_major_loader:
            frames  = frames.to(device)
            outputs = model_major(frames)
            preds   = outputs.argmax(1).cpu().numpy()
            all_pred_major.extend(preds)
            all_true_major.extend(labels.numpy())

    true_major = np.array(all_true_major)
    pred_major = np.array(all_pred_major)

    major_class_names = [str(idx2name[major_idx2cls[i]]) for i in range(len(majority_classes))]

    cm_major = confusion_matrix(true_major, pred_major)
    plot_confusion_matrix(cm_major, major_class_names, "Confusion Matrix — MAJOR Model")

    print("\n===== MAJOR Model Report =====")
    print(classification_report(true_major, pred_major,
                                target_names=major_class_names,
                                zero_division=0))

    # --- Few-shot evaluation ---
    all_true_fs, all_pred_fs = [], []

    with torch.no_grad():
        for frames, labels in val_fewshot_loader:
            frames = frames.to(device)
            embs   = embedder(frames)
            for emb, lbl in zip(embs, labels):
                pred_cls, _ = matcher.predict(emb.cpu())
                all_pred_fs.append(pred_cls)
                all_true_fs.append(lbl.item())

    true_fs = np.array(all_true_fs)
    pred_fs = np.array(all_pred_fs)

    present_labels     = sorted(np.unique(np.concatenate([true_fs, pred_fs])))
    fs_class_names     = [str(idx2name[fewshot_idx2cls[i]]) for i in present_labels]

    cm_fs = confusion_matrix(true_fs, pred_fs, labels=present_labels)
    plot_confusion_matrix(cm_fs, fs_class_names, "Confusion Matrix — Few-Shot Model")

    print("\n===== Few-Shot Model Report =====")
    print(classification_report(true_fs, pred_fs,
                                labels=present_labels,
                                target_names=fs_class_names,
                                zero_division=0))

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()