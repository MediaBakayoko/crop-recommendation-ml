"""
01_eda.py  -  Analyse exploratoire du Crop Recommendation Dataset (sol & climat).
Sortie : reports/figures/distributions.png, correlation_heatmap.png, boxplots_by_crop.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "raw", "crop_recommendation.csv")
FIG_DIR = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NUM_COLS = ['N', 'P', 'K', 'Cu', 'Fe', 'Mg', 'S', 'temperature', 'humidity', 'ph', 'rainfall']


def load():
    df = pd.read_csv(DATA_PATH)
    df.columns = NUM_COLS + ['label']
    return df


def run():
    df = load()
    print(f"Dataset : {df.shape[0]} observations, {df.shape[1]} variables")
    print("Valeurs manquantes :\n", df.isna().sum())
    print("Doublons :", df.duplicated().sum())
    print("Répartition des cultures :\n", df['label'].value_counts())

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, col in zip(axes.flat, NUM_COLS):
        sns.histplot(df[col], kde=True, ax=ax, color="teal")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "distributions.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[NUM_COLS].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matrice de corrélation  -  variables sol/climat")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, col in zip(axes.flat, ['N', 'temperature', 'humidity', 'rainfall']):
        order = df.groupby('label')[col].median().sort_values().index
        sns.boxplot(data=df, x='label', y=col, order=order, ax=ax)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_title(f"{col} par culture")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boxplots_by_crop.png"), dpi=130)
    plt.close()
    print("Figures EDA sauvegardées.")
    return df


if __name__ == "__main__":
    run()
