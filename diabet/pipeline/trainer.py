
# pipeline/trainer.py

import pandas as pd
from core.model import DiabetesModel, FEATURES
from core.dataset import generate_dataset


class Trainer:
    def __init__(self):
        # Création du modèle de prédiction du diabète
        self.model = DiabetesModel()

    def train(self, data_path):
        print("--- Début de l'entraînement (Diabète) ---")

        # Étape 1 : Générer et sauvegarder le dataset
        generate_dataset(data_path)

        # Étape 2 : Charger les données depuis le fichier CSV
        df = pd.read_csv(data_path)

        # Étape 3 : Vérifier la présence des colonnes nécessaires
        missing_columns = [
            col for col in FEATURES if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans le dataset : {missing_columns}"
            )

        if "diabetes" not in df.columns:
            raise ValueError(
                "La colonne cible 'diabetes' est absente du dataset."
            )

        # Étape 4 : Séparer les caractéristiques et la cible
        X = df[FEATURES]
        y = df["diabetes"]

        # Étape 5 : Entraîner le modèle
        self.model.fit(X, y)

        print("--- Entraînement terminé avec succès ---")

        # Étape 6 : Retourner le modèle entraîné
        return self.model
