# mlflow_config.py
import mlflow
import os

def setup_mlflow(experiment_name: str = "Drug_Classification_Experiment"):
    """
    Configure MLflow pour le projet.
    """
    import mlflow

    # Définir le tracking URI (serveur SQLite)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # On ne force pas artifact_location
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Expérience créée : {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Expérience existante : {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        print(f"❌ Erreur lors de la configuration MLflow : {e}")
        raise



def log_dataset_info(df, X_train, X_test, y_train, y_test):
    """
    Log les informations sur le dataset.
    """
    mlflow.log_param("dataset_total_rows", len(df))
    mlflow.log_param("dataset_features", df.shape[1])
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("test_ratio", len(X_test) / (len(X_train) + len(X_test)))
    
    # Distribution des classes
    class_distribution = y_train.value_counts().to_dict()
    for drug, count in class_distribution.items():
        mlflow.log_param(f"class_{drug}_count", count)


def log_model_params(model):
    """
    Log les hyperparamètres du modèle.
    """
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for param_name, param_value in params.items():
            mlflow.log_param(f"model_{param_name}", param_value)


def log_metrics(y_test, y_pred, accuracy):
    """
    Log les métriques d'évaluation.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Métriques principales
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_macro", precision_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("recall_macro", recall_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average='macro'))
    
    # Métriques weighted
    mlflow.log_metric("precision_weighted", precision_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("recall_weighted", recall_score(y_test, y_pred, average='weighted'))
    mlflow.log_metric("f1_weighted", f1_score(y_test, y_pred, average='weighted'))


def log_confusion_matrix(y_test, y_pred):
    """
    Log la matrice de confusion comme artefact.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    # Sauvegarder et logger
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # Nettoyer le fichier temporaire
    if os.path.exists("confusion_matrix.png"):
        os.remove("confusion_matrix.png")


def log_classification_report(y_test, y_pred):
    """
    Log le rapport de classification comme artefact texte.
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_test, y_pred)
    
    # Sauvegarder dans un fichier temporaire
    with open("classification_report.txt", "w") as f:
        f.write(report)
    
    mlflow.log_artifact("classification_report.txt")
    
    # Nettoyer
    if os.path.exists("classification_report.txt"):
        os.remove("classification_report.txt")
