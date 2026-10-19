
from sklearn.linear_model import LogisticRegression

# Les caractéristiques utilisées pour prédire le diabète
FEATURES = [
    "pregnancies",
    "glucose",
    "bloodpressure",
    "skin thickness",
    "insulin",
    "bmi",
    "diabetes pedigree function",
    "age"
]


class DiabetesModel:

    def __init__(self):
        # Modèle de régression logistique
        self.model = LogisticRegression(max_iter=1000)
        self.features = FEATURES

    def fit(self, X, y):
        """Entraîne le modèle sur les données d'entraînement."""
        self.model.fit(X, y)
        print(f"Modèle entraîné sur {len(X)} patients.")

    def predict(self, features):
        """Retourne la probabilité prédite de diabète."""
        proba = self.model.predict_proba([features])[0][1]
        return proba
