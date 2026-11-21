"""
03_interpretability.py  -  Interprétabilité du modèle Random Forest via SHAP.
Sortie : reports/figures/shap_importance.png, reports/shap_summary.csv
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "raw", "crop_recommendation.csv")
MODEL_DIR = os.path.join(BASE, "app", "model")
REPORT_DIR = os.path.join(BASE, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

NUM_COLS = ['N', 'P', 'K', 'Cu', 'Fe', 'Mg', 'S', 'temperature', 'humidity', 'ph', 'rainfall']


def run(sample_size=200):
    import shap
    df = pd.read_csv(DATA_PATH)
    df.columns = NUM_COLS + ['label']
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X = df[NUM_COLS]
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    shap_importance = pd.Series(mean_abs_shap, index=NUM_COLS).sort_values(ascending=False)
    shap_importance.to_csv(os.path.join(REPORT_DIR, "shap_summary.csv"))
    print(shap_importance)

    plt.figure(figsize=(8, 5))
    shap_importance.plot(kind="barh", color="darkorange")
    plt.gca().invert_yaxis()
    plt.title("Importance SHAP moyenne (toutes cultures)  -  Random Forest")
    plt.xlabel("|valeur SHAP| moyenne")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_importance.png"), dpi=150)
    plt.close()
    return shap_importance


if __name__ == "__main__":
    run()
