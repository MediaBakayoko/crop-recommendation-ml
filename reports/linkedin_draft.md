# Post LinkedIn  -  brouillon en français (à valider)

🌱 LE MACHINE LEARNING PEUT-IL NOUS AIDER À MIEUX DÉCIDER EN AGRICULTURE ?

Je viens de terminer un projet de Machine Learning appliqué à l'agriculture de précision.

🎯 Objectif : développer un modèle capable de recommander la culture la plus adaptée à partir des
caractéristiques du sol (azote, phosphore, potassium, cuivre, fer, magnésium, soufre, pH) et des
paramètres climatiques (température, humidité, précipitations).

Avec Python et Scikit-learn, j'ai exploré 2 200 observations réparties sur 22 cultures et comparé
trois algorithmes de classification :
• Arbre de décision  -  93,9% de précision
• Forêt aléatoire (Random Forest)  -  100% de précision (99,9% en validation croisée)
• K plus proches voisins (KNN)  -  97,5% de précision

📊 L'interprétabilité du modèle (Feature Importance + SHAP) converge vers la même conclusion : les
précipitations, l'humidité et le potassium sont les facteurs les plus déterminants  -  le climat et
la fertilisation comptent davantage que les micronutriments secondaires.

💻 J'ai aussi conçu un tableau de bord interactif (Streamlit) : on saisit les paramètres sol/climat
et on obtient instantanément la culture recommandée avec un score de confiance.

⚠️ Précision scientifique importante : ce jeu de données est construit à partir de données
agricoles indiennes. C'est un exercice méthodologique, pas un modèle validé pour les conditions
agro-climatiques ivoiriennes ou ouest-africaines  -  un déploiement réel nécessiterait un
réentraînement sur des données locales (ANADER, CNRA, SODEXAM). Assumer cette limite me semble
tout aussi important que la performance du modèle elle-même.

Mon objectif reste de développer des solutions concrètes à la croisée de l'Agriculture, des
Sciences Géospatiales, de la Statistique et de l'Intelligence Artificielle. Parce que l'avenir de
l'agriculture africaine dépendra aussi de notre capacité à comprendre et exploiter la donnée  -  avec
rigueur.

💻 Code complet, méthodologie et tableau de bord en open-source :
🔗 https://github.com/MediaBakayoko/crop-recommendation-ml

#MachineLearning #AgriDataScience #AgriculturePrecision #Python #DataScience
#IntelligenceArtificielle #Afrique #GeoAI
