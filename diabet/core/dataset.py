
import pandas as pd
from sklearn.datasets import fetch_openml
import os


def generate_dataset(filepath):
    """
    Charge le dataset Pima Indians Diabetes depuis OpenML
    et le sauvegarde dans un fichier CSV.

    Cible :
        0 = Non diabétique
        1 = Diabétique
    """

    # Créer le dossier de destination si nécessaire
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Chargement du dataset depuis OpenML
    data = fetch_openml(
        name="diabetes",
        version=1,
        as_frame=True
    )

    # Récupération des caractéristiques
    df = data.data.copy()

    # Ajout de la colonne cible
    df["diabetes"] = data.target

    # Normaliser les étiquettes en 0 et 1
    df["diabetes"] = (
        df["diabetes"]
        .astype(str)
        .str.strip()
        .map({
            "tested_negative": 0,
            "tested_positive": 1,
            "0": 0,
            "1": 1
        })
    )

    # Vérifier que la cible a bien été convertie
    if df["diabetes"].isna().any():
        raise ValueError(
            "Impossible de convertir certaines valeurs de la cible."
        )

    df["diabetes"] = df["diabetes"].astype(int)

    # Sauvegarde du dataset
    df.to_csv(filepath, index=False)

    print(f"Dataset Diabète généré : {filepath}")
    print(f"   - {len(df)} patients")
    print(f"   - {len(df.columns) - 1} features")
    print(f"   - Diabétiques : {(df['diabetes'] == 1).sum()}")
    print(f"   - Non diabétiques : {(df['diabetes'] == 0).sum()}")

    return df


if __name__ == "__main__":
    generate_dataset("data/diabetes_data.csv")
