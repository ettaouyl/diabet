
# app/interface_clinique.py

class ClinicalPredictor:
    def __init__(self, trained_model):
        # Récupérer le modèle déjà entraîné
        self.model = trained_model

        # Récupérer les caractéristiques attendues par le modèle
        self.features = trained_model.features

    def diagnose(self, patient_data):
        """
        Estime la probabilité de diabète d'un patient.

        patient_data est un dictionnaire contenant les valeurs
        des caractéristiques du patient.

        Exemple :
        {
            'preg': 2,
            'plas': 120,
            'pres': 70,
            'skin': 25,
            'insu': 80,
            'mass': 28.5,
            'pedi': 0.35,
            'age': 35
        }
        """

        # Étape 1 : Vérifier que toutes les données sont présentes
        missing_features = [
            feature for feature in self.features
            if feature not in patient_data
        ]

        if missing_features:
            raise ValueError(
                f"Données manquantes : {missing_features}"
            )

        # Étape 2 : Organiser les données dans le bon ordre
        features_list = [
            float(patient_data[feature])
            for feature in self.features
        ]

        # Étape 3 : Calculer la probabilité prédite
        proba = self.model.predict(features_list)

        # Étape 4 : Afficher le résultat
        print(f"Probabilité prédite de diabète : {proba:.2%}")

        # Étape 5 : Retourner une classe prédite
        if proba >= 0.5:
            return "Risque élevé selon le modèle (classe prédite : 1)"
        else:
            return "Risque faible selon le modèle (classe prédite : 0)"
