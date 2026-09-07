# model_pipeline_mlflow.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow_config import (
    setup_mlflow, log_dataset_info, log_model_params, 
    log_metrics, log_confusion_matrix, log_classification_report
)

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


def train_model_with_mlflow(X_train, y_train, max_iter=1000, C=1.0, solver='lbfgs', random_state=42):
    """
    Entraîne un modèle Logistic Regression avec tracking MLflow.
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        max_iter: Nombre maximum d'itérations
        C: Inverse de la force de régularisation
        solver: Algorithme d'optimisation
        random_state: Seed pour la reproductibilité
    
    Returns:
        model entraîné
    """
    print("Entraînement du modèle Logistic Regression avec MLflow...")
    
    # Créer le modèle avec les hyperparamètres
    model = LogisticRegression(
        max_iter=max_iter, 
        C=C, 
        solver=solver,
        random_state=random_state
    )
    
    # Entraîner
    model.fit(X_train, y_train)
    
    print("Modèle entraîné avec succès!")
    return model


def evaluate_model_with_mlflow(model, X_test, y_test):
    """
    Évalue le modèle et log les métriques dans MLflow.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\n=== Évaluation du Modèle ===")
    print(f"Accuracy : {accuracy:.4f}")
    print("\nClassification Report :\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix :\n", confusion_matrix(y_test, y_pred))
    
    return accuracy, y_pred


def run_mlflow_experiment(
    data_path: str = "data/drug200.csv",
    max_iter: int = 1000,
    C: float = 1.0,
    solver: str = 'lbfgs',
    experiment_name: str = "Drug_Classification_Experiment",
    run_name: str = None
):
    """
    Exécute une expérience complète avec tracking MLflow.
    
    Args:
        data_path: Chemin vers le fichier CSV
        max_iter: Nombre maximum d'itérations
        C: Inverse de la force de régularisation
        solver: Algorithme d'optimisation
        experiment_name: Nom de l'expérience MLflow
        run_name: Nom du run (optionnel)
    """
    # Configuration MLflow
    setup_mlflow(experiment_name)
    
    # Démarrer un run MLflow
    with mlflow.start_run(run_name=run_name):
        print("\n🚀 Démarrage de l'expérience MLflow...")
        
        # 1. Préparation des données
        print("\n📊 Étape 1: Préparation des données")
        X_train, X_test, y_train, y_test, df = prepare_data(data_path)
        
        # Log des informations sur le dataset
        log_dataset_info(df, X_train, X_test, y_train, y_test)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("random_state", 42)
        
        # 2. Entraînement du modèle
        print("\n🎯 Étape 2: Entraînement du modèle")
        model = train_model_with_mlflow(X_train, y_train, max_iter, C, solver, 42)
        
        # Log des hyperparamètres
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        log_model_params(model)
        
        # 3. Évaluation
        print("\n📈 Étape 3: Évaluation du modèle")
        accuracy, y_pred = evaluate_model_with_mlflow(model, X_test, y_test)
        
        # Log des métriques
        log_metrics(y_test, y_pred, accuracy)
        
        # Log des artefacts (confusion matrix, classification report)
        print("\n📁 Étape 4: Enregistrement des artefacts")
        log_confusion_matrix(y_test, y_pred)
        log_classification_report(y_test, y_pred)
        
        # 4. Enregistrement du modèle
        print("\n💾 Étape 5: Enregistrement du modèle")
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="Drug_Classification_LogisticRegression"
        )
        
        # Sauvegarder aussi localement avec joblib
        model_path = "models/logistic_regression_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"Modèle sauvegardé localement : {model_path}")
        
        # Log des tags pour faciliter la recherche
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("dataset", "drug200")
        mlflow.set_tag("task", "multiclass_classification")
        
        print("\n✅ Expérience MLflow terminée avec succès!")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Run ID: {mlflow.active_run().info.run_id}")
        print(f"   - Experiment ID: {mlflow.active_run().info.experiment_id}")
        
        return model, accuracy


def load_model_from_mlflow(run_id: str):
    """
    Charge un modèle depuis MLflow en utilisant le run_id.
    
    Args:
        run_id: ID du run MLflow
    
    Returns:
        Modèle chargé
    """
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modèle chargé depuis MLflow (Run ID: {run_id})")
    return model


def load_model(model_path: str = "models/logistic_regression_model.joblib"):
    """
    Charge un modèle précédemment sauvegardé (méthode classique).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
    
    model = joblib.load(model_path)
    print(f"Modèle chargé depuis : {model_path}")
    return model
