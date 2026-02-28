# 🫒 Olive Leaf Disease Detector

Application web de détection automatique des maladies des feuilles d'olivier.

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure du projet

```
olive_app/
├── app.py                  # Application Streamlit
├── requirements.txt        # Dépendances
├── models/
│   ├── svm_olive_pipeline.pkl   # Modèle SVM entraîné
│   └── label_encoder.pkl        # Encodeur des labels
└── README.md
```

## Déploiement sur Streamlit Cloud

1. Crée un repo GitHub et pousse ces fichiers
2. Va sur https://share.streamlit.io
3. Connecte ton repo GitHub
4. Sélectionne `app.py` comme fichier principal
5. Clique Deploy !

## Classes détectées

| Classe | Description |
|--------|-------------|
| Healthy | Feuille saine |
| Aculus Olearius | Acarien microscopique |
| Olive Peacock Spot | Maladie fongique (Cycloconium) |

## Modèle

- Algorithme : SVM (kernel RBF, C=10)
- Features : Pixels aplatis + Histogramme RGB (64×64)
- Accuracy test : **78.68%**
