"""
app/dashboard.py — Tableau de bord Streamlit pour la recommandation de culture.
Lancer avec : streamlit run app/dashboard.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

st.set_page_config(page_title="Recommandation de culture", page_icon="🌱", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, le, feature_names

model, le, feature_names = load_artifacts()

st.title("🌱 Recommandation de culture par Machine Learning")
st.markdown(
    "Renseignez les caractéristiques du sol et du climat pour obtenir la culture la plus adaptée, "
    "prédite par un modèle **Random Forest** entraîné sur le *Crop Recommendation Dataset* "
    "(2 200 observations, 22 cultures)."
)
st.warning(
    "⚠️ Ce jeu de données est construit à partir de données agricoles indiennes. "
    "Il s'agit d'un exercice pédagogique de démonstration méthodologique — les seuils "
    "ne sont pas directement validés pour les conditions agro-climatiques ivoiriennes."
)

col1, col2 = st.columns(2)
with col1:
    N = st.slider("Azote - N (kg/ha)", 0, 140, 50)
    P = st.slider("Phosphore - P (kg/ha)", 5, 145, 50)
    K = st.slider("Potassium - K (kg/ha)", 5, 205, 50)
    Cu = st.slider("Cuivre - Cu", 0.0, 5.0, 1.0)
    Fe = st.slider("Fer - Fe", 0.0, 10.0, 2.0)
    Mg = st.slider("Magnésium - Mg", 0.0, 10.0, 2.0)
with col2:
    S = st.slider("Soufre - S", 0.0, 10.0, 2.0)
    temperature = st.slider("Température (°C)", 0.0, 45.0, 25.0)
    humidity = st.slider("Humidité (%)", 0.0, 100.0, 60.0)
    ph = st.slider("pH du sol", 3.0, 10.0, 6.5)
    rainfall = st.slider("Précipitations (mm)", 0.0, 300.0, 100.0)

if st.button("🔍 Prédire la culture recommandée", type="primary"):
    X = pd.DataFrame([[N, P, K, Cu, Fe, Mg, S, temperature, humidity, ph, rainfall]],
                      columns=feature_names)
    pred = model.predict(X)[0]
    crop = le.inverse_transform([pred])[0]
    proba = model.predict_proba(X)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], proba[i]) for i in top3_idx]

    st.success(f"🌾 Culture recommandée : **{crop.upper()}**")
    st.markdown("**Top 3 des cultures les plus probables :**")
    for name, p in top3:
        st.progress(float(p), text=f"{name} — {p*100:.1f}%")

st.markdown("---")
st.caption(
    "Projet : *Machine Learning for Smart Agriculture: Predicting the Most Suitable Crop "
    "Based on Soil and Climatic Conditions* — code source sur GitHub."
)
