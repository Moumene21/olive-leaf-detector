import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import gdown

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olive Leaf Disease Detector",
    page_icon="🫒",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1a0f; color: #e8f0e8; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; color: #a8d5a2;
    text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.5px;
}
.hero-sub {
    text-align: center; color: #6b8f6b;
    font-size: 1rem; font-weight: 300; margin-bottom: 2rem;
}
.result-card {
    background: #1a2e1a; border-radius: 16px;
    padding: 1.5rem 2rem; margin-top: 1.5rem;
    border-left: 5px solid #a8d5a2;
}
.result-label { font-family: 'Playfair Display', serif; font-size: 1.8rem; color: #a8d5a2; margin: 0; }
.result-conf { color: #6b8f6b; font-size: 0.95rem; margin-top: 0.3rem; }
.disease-badge {
    display: inline-block; padding: 0.3rem 1rem;
    border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin-top: 0.5rem;
}
.healthy { background: #1e4d2b; color: #6fcf97; }
.disease { background: #4d1e1e; color: #eb5757; }
.divider { border: none; border-top: 1px solid #2a3f2a; margin: 1.5rem 0; }
.info-box {
    background: #152015; border-radius: 12px;
    padding: 1rem 1.5rem; margin-top: 1rem;
    font-size: 0.88rem; color: #8aab8a; line-height: 1.6;
}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Téléchargement automatique des modèles depuis Google Drive ────────────────
@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)

    pipeline_path = "models/svm_olive_pipeline.pkl"
    le_path       = "models/label_encoder.pkl"

    if not os.path.exists(pipeline_path):
        with st.spinner("Téléchargement du modèle SVM..."):
            gdown.download(
                "https://drive.google.com/uc?id=1wSFUTu1IKtaR2k71CAsTyce16mGelDcU",
                pipeline_path, quiet=False
            )

    if not os.path.exists(le_path):
        with st.spinner("Téléchargement du LabelEncoder..."):
            gdown.download(
                "https://drive.google.com/uc?id=1jSb4lJmZj-W-TXdHQvW-hMH1rug1e51o",
                le_path, quiet=False
            )

    pipeline = joblib.load(pipeline_path)
    le       = joblib.load(le_path)
    return pipeline, le


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img: Image.Image, size=(64, 64)):
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    pixels = arr.flatten()
    hist_r, _ = np.histogram(arr[:,:,0], bins=256, range=(0,1))
    hist_g, _ = np.histogram(arr[:,:,1], bins=256, range=(0,1))
    hist_b, _ = np.histogram(arr[:,:,2], bins=256, range=(0,1))
    histogram  = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    histogram /= (histogram.sum() + 1e-8)
    return np.concatenate([pixels, histogram])


# ── Infos par classe ──────────────────────────────────────────────────────────
CLASS_INFO = {
    "Healthy": {
        "fr": "Feuille Saine", "badge": "healthy",
        "desc": "La feuille ne présente aucun signe de maladie. La plante est en bonne santé.",
        "action": "Aucune action nécessaire. Continuez le suivi régulier.",
        "icon": "🌿"
    },
    "aculus_olearius": {
        "fr": "Aculus Olearius", "badge": "disease",
        "desc": "Infestation par un acarien microscopique (Aculus olearius) causant des déformations foliaires et une réduction du rendement.",
        "action": "Traitement acaricide recommandé. Consultez un agronome.",
        "icon": "🔴"
    },
    "olive_peacock_spot": {
        "fr": "Tache Paon (Cycloconium)", "badge": "disease",
        "desc": "Maladie fongique causée par Spilocaea oleagina, reconnaissable à ses taches circulaires brun-verdâtres.",
        "action": "Traitement fongicide au cuivre recommandé en automne.",
        "icon": "🟠"
    }
}


# ── Interface ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🫒 Olive Leaf Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Détection automatique des maladies des feuilles d\'olivier par SVM</p>', unsafe_allow_html=True)

pipeline, le = load_model()

uploaded = st.file_uploader(
    "Glissez une image de feuille d'olivier",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is not None:
    img = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, caption="Image uploadée", use_container_width=True)

    with col2:
        with st.spinner("Analyse en cours..."):
            features   = extract_features(img).reshape(1, -1)
            pred_idx   = pipeline.predict(features)[0]
            pred_proba = pipeline.predict_proba(features)[0]
            pred_class = le.classes_[pred_idx]
            confidence = pred_proba[pred_idx] * 100

        info = CLASS_INFO[pred_class]

        st.markdown(f"""
        <div class="result-card">
            <p style="font-size:2rem; margin:0">{info['icon']}</p>
            <p class="result-label">{info['fr']}</p>
            <p class="result-conf">Confiance : <strong>{confidence:.1f}%</strong></p>
            <span class="disease-badge {info['badge']}">
                {'Saine' if info['badge'] == 'healthy' else 'Maladie détectée'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <strong>Description</strong><br>{info['desc']}
            <br><br>
            <strong>Action recommandée</strong><br>{info['action']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Probabilités par classe**")
    for i, cls in enumerate(le.classes_):
        prob  = pred_proba[i] * 100
        color = "#a8d5a2" if cls == pred_class else "#3a5c3a"
        st.markdown(f"""
        <div style="margin-bottom:0.5rem">
            <div style="display:flex; justify-content:space-between;
                        font-size:0.85rem; color:#8aab8a; margin-bottom:3px">
                <span>{CLASS_INFO[cls]['fr']}</span><span>{prob:.1f}%</span>
            </div>
            <div style="background:#1a2e1a; border-radius:20px; height:8px">
                <div style="background:{color}; width:{prob}%;
                            height:8px; border-radius:20px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#1a2e1a; border:2px dashed #3a5c3a; border-radius:16px;
                padding:2rem; text-align:center">
        <p style="font-size:2.5rem; margin:0">🍃</p>
        <p style="color:#6b8f6b; margin:0.5rem 0 0">
            Uploadez une image JPG ou PNG d'une feuille d'olivier
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; gap:1rem">
    <div style="background:#152015; border-radius:10px; padding:0.8rem 1.2rem; flex:1; text-align:center">
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; color:#a8d5a2">78.68%</div>
        <div style="font-size:0.75rem; color:#5a7a5a; margin-top:0.2rem">Accuracy (Test)</div>
    </div>
    <div style="background:#152015; border-radius:10px; padding:0.8rem 1.2rem; flex:1; text-align:center">
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; color:#a8d5a2">SVM</div>
        <div style="font-size:0.75rem; color:#5a7a5a; margin-top:0.2rem">Algorithme (kernel RBF)</div>
    </div>
    <div style="background:#152015; border-radius:10px; padding:0.8rem 1.2rem; flex:1; text-align:center">
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; color:#a8d5a2">3</div>
        <div style="font-size:0.75rem; color:#5a7a5a; margin-top:0.2rem">Classes détectées</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align:center; color:#3a5c3a; font-size:0.8rem; margin-top:2rem">
    Projet Master 1 Intelligence Artificielle · Université de Béjaïa
</p>
""", unsafe_allow_html=True)
