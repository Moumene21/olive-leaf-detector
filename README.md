# 🫒 Olive Leaf Disease Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

**Automatic detection of olive leaf diseases using Machine Learning (SVM)**

[🚀 Live Demo](https://olive-leaf-detector-z9atjfjs4qiayuyraappin3.streamlit.app) · [📓 Notebook](olive_app/Olive_SVM_RF.ipynb) · [📊 Dataset](https://www.kaggle.com/datasets/vineethakkinapalli/olive-leaf-disease-dataset)

</div>

---

## 🖥️ Application Screenshots

<div align="center">

### Main Interface — Healthy Leaf Detection
![App Main](screenshots/screenshot_app_main.png)

### Prediction — Aculus Olearius (87.7% confidence)
![App Aculus](screenshots/screenshot_aculus.png)

### Prediction — Tache Paon / Peacock Spot (93.2% confidence)
![App Peacock](screenshots/screenshot_peacock.png)

### Probability Scores per Class
![App Probabilities](screenshots/screenshot_probabilities.png)

</div>

---

## 📌 Overview

This project implements an **automatic classification system** for olive leaf diseases using classical Machine Learning algorithms. Given an image of an olive leaf, the system predicts whether the leaf is healthy or affected by one of two diseases.

Developed as part of the **Master 1 Artificial Intelligence** curriculum at the **University of Béjaïa, Algeria**.

---

## 🎯 Detected Classes

| Class | Description | Severity |
|-------|-------------|----------|
| 🌿 **Healthy** | No signs of disease | — |
| 🔴 **Aculus Olearius** | Microscopic mite infestation causing leaf deformation | High |
| 🟠 **Olive Peacock Spot** | Fungal disease (*Spilocaea oleagina*) with circular brown spots | Medium |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — Olive Leaf Disease Dataset |
| Total images | 2,720 |
| Healthy | 830 images (30.5%) |
| Aculus Olearius | 690 images (25.4%) |
| Peacock Spot | 1,200 images (44.1%) |
| Image format | JPG / PNG |

---

## 🧠 Machine Learning Pipeline

```
Raw Images
    │
    ▼
Feature Extraction (64×64)
    ├── Flattened pixels  → 12,288 features
    └── RGB histogram     →    768 features
                               ─────────────
                          Total: 13,056 features
    │
    ▼
StandardScaler (normalization)
    │
    ▼
SVM Classifier (kernel RBF, C=10)
    │
    ▼
Prediction + Confidence Score
```

---

## 📈 Results

| Algorithm | Test Accuracy | Training Time |
|-----------|:-------------:|:-------------:|
| **SVM (RBF, C=10)** | **78.68%** ✅ | ~5 min |
| Random Forest (300 trees) | 75.49% | ~2 min |

> The **SVM** was selected as the final model due to its superior accuracy.

### Algorithm Comparison
![Comparison](screenshots/screenshot_comparison.png)

---

## 🔵 Confusion Matrices

<div align="center">

### SVM — Test Set
![SVM Confusion Matrix](screenshots/screenshot_cm_svm.png)

### Random Forest — Test Set
![RF Confusion Matrix](screenshots/screenshot_cm_rf.png)

</div>

---

## 🖼️ SVM Predictions on Test Set

![Predictions](screenshots/screenshot_predictions.png)

---

## 🗂️ Project Structure

```
olive-leaf-detector/
├── olive_app/
│   ├── app.py                  # Streamlit web application
│   ├── requirements.txt        # Python dependencies
│   └── Olive_SVM_RF.ipynb      # Complete notebook (EDA + SVM + RF)
├── screenshots/                # Project screenshots
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Moumene21/olive-leaf-detector.git
cd olive-leaf-detector/olive_app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the app**
```bash
streamlit run app.py
```

> **Note:** Models are automatically downloaded from Google Drive on first launch.

---

## 🌐 Live Demo

**🔗 https://olive-leaf-detector-z9atjfjs4qiayuyraappin3.streamlit.app**

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| `Python 3.13` | Core language |
| `scikit-learn 1.8.0` | SVM, Random Forest, preprocessing |
| `NumPy` | Numerical computation |
| `Pillow` | Image processing |
| `Streamlit` | Web interface |
| `joblib` | Model serialization |
| `gdown` | Model download from Google Drive |
| `matplotlib / seaborn` | Data visualization |

---

## 👤 Author

**Moumene** — Master 1 Artificial Intelligence
University of Béjaïa, Algeria

[![GitHub](https://img.shields.io/badge/GitHub-Moumene21-black?style=flat-square&logo=github)](https://github.com/Moumene21)

---

## 📄 License

This project is licensed under the MIT License.
