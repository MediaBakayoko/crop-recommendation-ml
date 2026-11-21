"""
02_modeling.py  -  Entraînement et comparaison de Decision Tree, Random Forest et KNN
pour la recommandation de culture à partir des variables sol/climat.

Sortie : reports/classification_report.txt, reports/figures/confusion_matrices.png,
         reports/figures/feature_importance.png, app/model/*.pkl
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report)

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "raw", "crop_recommendation.csv")
REPORT_DIR = os.path.join(BASE, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
MODEL_DIR = os.path.join(BASE, "app", "model")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_COLS = ['N', 'P', 'K', 'Cu', 'Fe', 'Mg', 'S', 'temperature', 'humidity', 'ph', 'rainfall']


def run():
    df = pd.read_csv(DATA_PATH)
    df.columns = NUM_COLS + ['label']

    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X = df[NUM_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results, preds = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        Xtr = X_train_s if name == "KNN" else X_train
        Xte = X_test_s if name == "KNN" else X_test
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        preds[name] = pred
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="macro"),
            "Recall": recall_score(y_test, pred, average="macro"),
            "F1": f1_score(y_test, pred, average="macro"),
            "CV_accuracy_mean": cv_scores.mean(),
            "CV_accuracy_std": cv_scores.std(),
        }
        print(f"=== {name} === {results[name]}")

    # Matrices de confusion
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, (name, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Matrice de confusion  -  {name}")
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=130)
    plt.close()

    # Meilleur modèle = Random Forest (comparaison empirique, voir rapport)
    best_name = max(results, key=lambda n: results[n]["Accuracy"])
    best_model = models[best_name]
    report_txt = classification_report(y_test, preds[best_name], target_names=le.classes_)

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== Comparaison des 3 modèles ===\n\n")
        for name, r in results.items():
            f.write(f"{name}: Accuracy={r['Accuracy']:.3f} Precision={r['Precision']:.3f} "
                    f"Recall={r['Recall']:.3f} F1={r['F1']:.3f} "
                    f"CV={r['CV_accuracy_mean']:.3f}±{r['CV_accuracy_std']:.3f}\n")
        f.write(f"\n=== Rapport détaillé  -  {best_name} (meilleur modèle) ===\n\n")
        f.write(report_txt)

    # Importance des variables (Random Forest)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=NUM_COLS).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        importances.plot(kind="barh", color="forestgreen")
        plt.gca().invert_yaxis()
        plt.title(f"Importance des variables  -  {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    # Sauvegarde du modèle pour le dashboard
    joblib.dump(best_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(NUM_COLS, os.path.join(MODEL_DIR, "feature_names.pkl"))

    print(f"\nMeilleur modèle : {best_name}")
    return results, best_model, le


if __name__ == "__main__":
    run()
