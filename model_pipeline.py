# model_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Créer le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)


def prepare_data(data_path: str = "data/drug200.csv"):
    """
    Charge et prétraite les données du fichier CSV.
    
    Returns:
        X_train, X_test, y_train, y_test, df (DataFrame original)
    """
    # Chargement des données
    df = pd.read_csv(data_path)
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Encodage des variables catégorielles
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()

    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['BP'] = le_bp.fit_transform(df['BP'])
    df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])

    # Séparation features / target
    X = df.drop('Drug', axis=1)
    y = df['Drug']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Données prêtes : Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test, df


def train_model(X_train, y_train):
    """
    Entraîne un modèle Logistic Regression.
    
    Returns:
        model entraîné
    """
    print("Entraînement du modèle Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès!")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle et affiche les métriques.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\n=== Évaluation du Modèle ===")
    print(f"Accuracy : {accuracy:.4f}")
    print("\nClassification Report :\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix :\n", confusion_matrix(y_test, y_pred))
    
    return accuracy


def save_model(model, model_path: str = "models/logistic_regression_model.joblib"):
    """
    Sauvegarde le modèle entraîné avec joblib.
    """
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans : {model_path}")


def load_model(model_path: str = "models/logistic_regression_model.joblib"):
    """
    Charge un modèle précédemment sauvegardé.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
    
    model = joblib.load(model_path)
    print(f"Modèle chargé depuis : {model_path}")
    return model