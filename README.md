# Machine Learning for Smart Agriculture: Predicting the Most Suitable Crop Based on Soil and Climatic Conditions

## Problématique
Comment utiliser les caractéristiques du sol et les paramètres climatiques pour identifier la
culture agricole la plus adaptée à un environnement donné ?

## Données
**Crop Recommendation Dataset**  -  2 200 observations, 22 cultures, variables de sol (N, P, K,
Cu, Fe, Mg, S, pH) et de climat (température, humidité, précipitations).

- Source principale : Kaggle  -  Atharva Ingle  - 
  https://www.kaggle.com/atharvaingle/crop-recommendation-dataset
- Version utilisée dans ce dépôt (11 variables, incluant Cu/Fe/Mg/S) : Mendeley Data (2025),
  licence CC BY 4.0  -  https://data.mendeley.com/datasets/7j8cz7p7w9
- Fichier local : `data/raw/crop_recommendation.csv`

> ⚠️ **Limite méthodologique importante** : ce jeu de données est construit à partir de données
> agricoles **indiennes**. Il s'agit d'un exercice pédagogique de démonstration méthodologique,
> pas d'un modèle directement validé pour les conditions agro-climatiques ivoiriennes ou
> ouest-africaines. Les seuils de sol/climat appris ici (ex. plages de pluviométrie, pH) reflètent
> le contexte indien et ne doivent pas être transposés tels quels. Voir la section « Limites ».

## Objectif
Développer un modèle de classification capable de prédire la culture la plus adaptée parmi
22 cultures, à partir des caractéristiques du sol et du climat.

## Méthodologie
1. **EDA** (`src/01_eda.py`) : distributions, corrélations, boxplots par culture.
2. **Prétraitement** : vérification valeurs manquantes/doublons (aucun constaté), standardisation
   pour KNN, encodage de la cible, split 80/20 stratifié.
3. **Modélisation** (`src/02_modeling.py`) : comparaison de Decision Tree, Random Forest et KNN,
   avec validation croisée stratifiée (5 folds) en complément du jeu de test.
4. **Évaluation** : Accuracy, Precision, Recall, F1-score (moyenne macro), matrices de confusion.
5. **Interprétabilité** (`src/03_interpretability.py`) : Feature Importance (Random Forest) et
   SHAP (valeurs moyennes absolues par variable, toutes classes confondues).
6. **Dashboard** (`app/dashboard.py`, Streamlit) : saisie interactive des paramètres sol/climat
   et prédiction en temps réel avec top-3 des cultures les plus probables.

## Résultats (voir `ARTICLE.md` pour l'interprétation complète)

| Modèle | Accuracy | Precision | Recall | F1 | CV accuracy (5-fold) |
|---|---|---|---|---|---|
| Decision Tree | 93,9% | 91,4% | 93,9% | 92,4% | 92,6% ± 1,9% |
| **Random Forest** | **100%** | **100%** | **100%** | **100%** | **99,9% ± 0,1%** |
| KNN (k=5) | 97,5% | 97,7% | 97,5% | 97,5% | 96,5% ± 0,7% |

Random Forest obtient une performance parfaite sur le jeu de test et quasi-parfaite en validation
croisée  -  confirmé par CV (pas un artefact de split), ce qui s'explique par la forte séparabilité
des classes dans ce dataset synthétique/semi-synthétique bien structuré.

Variables les plus discriminantes (Feature Importance + SHAP) : **humidité, potassium (K), azote
(N), phosphore (P), précipitations**.

## Reproductibilité
```bash
pip install -r requirements.txt
python src/01_eda.py
python src/02_modeling.py
python src/03_interpretability.py
streamlit run app/dashboard.py
```

## Article complet
Voir [`ARTICLE.md`](ARTICLE.md) pour l'analyse narrative complète avec interprétation des
résultats, graphiques et discussion des limites.

## Auteur
Média Marcel Bakayoko  -  Expert géomatique / données géospatiales & agricoles / GeoAI / Suivi-Évaluation.

## Licence des données
Crop Recommendation Dataset  -  CC BY 4.0 (Mendeley Data / Kaggle, données agricoles indiennes).
