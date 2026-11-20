# Post LinkedIn — brouillon à valider

🌱 CAN MACHINE LEARNING HELP US MAKE SMARTER AGRICULTURAL DECISIONS?

I recently completed a Machine Learning project focused on precision agriculture.

🎯 Objective: Develop a predictive model capable of recommending suitable crops based on soil
properties (N, P, K, Cu, Fe, Mg, S, pH) and climatic conditions (temperature, humidity, rainfall).

Using Python and Scikit-learn, I explored 2,200 observations across 22 crops and compared three
classification algorithms:
• Decision Tree — 93.9% accuracy
• Random Forest — 100% accuracy (99.9% in 5-fold cross-validation)
• K-Nearest Neighbors — 97.5% accuracy

📊 Interpretability (Feature Importance + SHAP) converged on the same conclusion: rainfall,
humidity and potassium are the strongest drivers of crop suitability — climate and fertilization
matter more than micronutrients in this dataset.

💻 I also built an interactive Streamlit dashboard where you input soil/climate parameters and get
an instant crop recommendation with confidence scores.

Beyond model accuracy, this project explores how data science can support agricultural
decision-making, resource optimization and sustainable farming.

⚠️ Important scientific caveat: this dataset is built from Indian agricultural data. It's a
methodological exercise, not a model validated for Ivorian or West African agro-climatic
conditions — a real-world deployment would require retraining on local soil and climate data
(ANADER, CNRA, SODEXAM). Being explicit about this limitation is, I think, as important as the
model's accuracy itself.

My goal is to keep developing practical solutions at the intersection of Agriculture, Geospatial
Science, Statistics and Artificial Intelligence. Because the future of African agriculture will
also depend on our ability to understand and leverage data — rigorously.

💻 Full code, methodology and dashboard open-source:
🔗 https://github.com/MediaBakayoko/crop-recommendation-ml

#MachineLearning #AgriDataScience #PrecisionAgriculture #Python #DataScience
#ArtificialIntelligence #Africa #GeoAI
