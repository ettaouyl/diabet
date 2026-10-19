
# utils/metrics.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred):
    """
    Évalue les prédictions du modèle.

    y_true : vraies classes.
    y_pred : classes prédites par le modèle.

    Retour :
        Dictionnaire contenant les métriques.
    """

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist()
    }

    return results


def display_metrics(results):
    """
    Affiche les métriques de manière lisible.
    """

    print("\n--- Évaluation du modèle diabète ---")
    print(f"Accuracy  : {results['accuracy']:.2%}")
    print(f"Précision : {results['precision']:.2%}")
    print(f"Rappel    : {results['recall']:.2%}")
    print(f"F1-score  : {results['f1_score']:.2%}")
    print("Matrice de confusion :")
    print(results["confusion_matrix"])
