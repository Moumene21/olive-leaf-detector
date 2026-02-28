# 🫒 Olive Leaf Disease Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

**Automatic detection of olive leaf diseases using Machine Learning (SVM)**

[🚀 Live Demo](https://olive-leaf-detector-z9atjfjs4qiayuyraappin3.streamlit.app) · [📓 Notebook](olive_app/Olive_SVM_RF.ipynb) · [📊 Dataset](https://www.kaggle.com/datasets/vineethakkinapalli/olive-leaf-disease-dataset)

![App Screenshot](https://img.shields.io/badge/Interface-Streamlit-darkgreen?style=flat-square)

</div>

---

## 📌 Overview

This project implements an **automatic classification system** for olive leaf diseases using classical Machine Learning algorithms. Given an image of an olive leaf, the system predicts whether the leaf is healthy or affected by one of two diseases.

The project was developed as part of the **Master 1 Artificial Intelligence** curriculum at the **University of Béjaïa, Algeria**.

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

---

## 🗂️ Project Structure

```
olive-leaf-detector/
├── olive_app/
│   ├── app.py                  # Streamlit web application
│   ├── requirements.txt        # Python dependencies
│   └── Olive_SVM_RF.ipynb      # Complete notebook (EDA + SVM + RF)
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

The app will open at `http://localhost:8501`

> **Note:** The models are automatically downloaded from Google Drive on first launch.

---

## 🌐 Deployment

The application is deployed on **Streamlit Cloud** and accessible at:

**🔗 https://olive-leaf-detector-z9atjfjs4qiayuyraappin3.streamlit.app**

---

## 🛠️ Tech Stack

| Tool | Usage |
|------|-------|
| `Python 3.13` | Core language |
| `scikit-learn` | SVM, Random Forest, preprocessing |
| `NumPy` | Numerical computation |
| `Pillow` | Image processing |
| `Streamlit` | Web interface |
| `joblib` | Model serialization |
| `gdown` | Model download from Google Drive |
| `matplotlib / seaborn` | Data visualization |

---

## 📓 Notebook Contents

The notebook `Olive_SVM_RF.ipynb` covers the full ML pipeline:

1. **EDA** — Class distribution, sample images, RGB channel analysis
2. **Feature Extraction** — Pixel flattening + RGB histogram
3. **Preprocessing** — Train/Val/Test split (70/15/15), StandardScaler
4. **SVM Training** — kernel RBF, C=10, probability=True
5. **Random Forest Training** — 300 trees, n_jobs=-1
6. **Evaluation** — Accuracy, classification report, confusion matrix
7. **Comparison** — SVM vs Random Forest bar chart

---

## 👤 Author

**Meddas Massinissa** — Master 1 Artificial Intelligence  
University of Béjaïa, Algeria

[![GitHub](https://img.shields.io/badge/GitHub-Moumene21-black?style=flat-square&logo=github)](https://github.com/Moumene21)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
