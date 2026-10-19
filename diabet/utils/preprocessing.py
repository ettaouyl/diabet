
# utils/preprocessing.py

import pandas as pd


def clean_data(df):
    """
    Nettoie le dataset des patients.

    Paramètre :
        df : DataFrame contenant les données des patients.

    Retour :
        DataFrame nettoyé.
    """

    # Supprimer les lignes dupliquées
    df = df.drop_duplicates().copy()

    # Remplacer les valeurs infinies par des valeurs manquantes
    df = df.replace([float("inf"), float("-inf")], float("nan"))

    # Remplacer les valeurs manquantes par la médiane
    for column in df.columns:
        if df[column].isna().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())

    return df


def split_features_target(df, features, target="diabetes"):
    """
    Sépare les caractéristiques X de la cible y.

    X : informations des patients.
    y : classe à prédire (0 ou 1).
    """

    missing = [
        col for col in features + [target]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Colonnes absentes : {missing}")

    X = df[features]
    y = df[target]

    return X, y
