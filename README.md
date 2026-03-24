# Action Recognition for Trauma THOMPSON T3 Challenge

## 📌 Overview

This project focuses on **action recognition in medical procedures** using egocentric (first-person) video data from the Trauma THOMPSON (T3) Challenge.

The goal is to automatically identify critical actions performed during life-saving interventions, enabling AI-assisted decision support in emergency and low-resource environments.

The challenge itself is designed to advance computer vision models for recognizing and predicting medical actions from real-world procedural videos. ([thompson-challenge.grand-challenge.org][1])

---

## 🎯 Objectives

* Recognize **current medical actions** from video frames
* Improve understanding of **procedural workflows**
* Support development of **AI-assisted emergency systems**

---

## 🧠 Problem Description

In the T3 Challenge, action recognition requires classifying actions from egocentric medical videos, often represented as:

* **Verb + Noun pairs** (e.g., *insert needle*, *apply tourniquet*)
* Evaluated using **Top-1 / Top-5 accuracy metrics** ([thompson-challenge.grand-challenge.org][2])

This task is particularly challenging due to:

* Unstructured environments
* High variability in procedures
* Occlusions and motion blur

---

## 🏗️ Pipeline

```text
Input Video
   ↓
Frame Extraction
   ↓
Preprocessing (resize, normalization)
   ↓
Feature Extraction
   ↓
Model (MLP / CNN / Transformer)
   ↓
Action Classification (Verb + Noun)
```

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Extract frames from videos
* Normalize and resize inputs
* Optional augmentation for robustness

### 2. Feature Extraction

* Spatial features from frames
* Temporal representation (if applicable)

### 3. Model Architecture

* Multi-Layer Perceptron (MLP) / Deep Learning model
* Learns mapping from features → action classes

### 4. Training

* Supervised learning using labeled action data
* Loss function: Cross-entropy
* Optimization: Adam / SGD

---

## 📊 Dataset

The dataset comes from the **Trauma THOMPSON Challenge**, which includes:

* Egocentric videos of life-saving procedures
* Multiple tasks including:

  * Action recognition
  * Action anticipation
  * Tool detection
  * Visual question answering ([t3challenge25.grand-challenge.org][3])

⚠️ Note: Dataset access requires registration and cannot be redistributed.

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/sSFlyingSs/Action-Recognition-for-Trauma-THOMPSON-T3-Challenge.git
cd Action-Recognition-for-Trauma-THOMPSON-T3-Challenge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python src/main.py
```

---

## 📈 Results

(Add your results here)

Example:

* Top-1 Accuracy: XX%
* Top-5 Accuracy: XX%

---

## 📁 Project Structure

```
.
├── src/
├── data/
├── models/
├── results/
├── diagrams/
└── README.md
```

---

## 🔬 Future Work

* Incorporate **temporal models** (e.g., LSTM, Transformer)
* Use **Video-based architectures** (e.g., SlowFast, Video Swin)
* Improve real-time inference performance
* Extend to **action anticipation**

---

## 📚 References

* Trauma THOMPSON Challenge (T3)
* MICCAI Challenge Proceedings
* Video Action Recognition literature

---

## 📜 License

(Choose a license: MIT / Apache 2.0 recommended)

---

## 🙌 Acknowledgements

This project is inspired by the Trauma THOMPSON Challenge, which aims to develop AI systems for assisting medical procedures in emergency scenarios. ([t3challenge25.grand-challenge.org][4])

[1]: https://thompson-challenge.grand-challenge.org/?utm_source=chatgpt.com "Overview - The Trauma THOMPSON Challenge - Grand Challenge"
[2]: https://thompson-challenge.grand-challenge.org/task-and-evaluation/?utm_source=chatgpt.com "Task And Evaluation - The Trauma THOMPSON Challenge - Grand Challenge"
[3]: https://t3challenge25.grand-challenge.org/submission-instructions/?utm_source=chatgpt.com "Submission Instructions - The Trauma THOMPSON Challenge 2025 - Grand Challenge"
[4]: https://t3challenge25.grand-challenge.org/t3challenge25/?utm_source=chatgpt.com "Overview - The Trauma THOMPSON Challenge 2025 - Grand Challenge"
