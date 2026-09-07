from mlflow_config import setup_mlflow, log_dataset_info, log_model_params, log_metrics, log_confusion_matrix, log_classification_report

# Configurer MLflow et récupérer l'ID de l'expérience
experiment_id = setup_mlflow("Drug_Classification_Experiment")

import mlflow
with mlflow.start_run(experiment_id=experiment_id):
    # Log du dataset
    log_dataset_info(df, X_train, X_test, y_train, y_test)

    # Log des paramètres du modèle
    log_model_params(model)

    # Prédictions et métriques
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    log_metrics(y_test, y_pred, accuracy)

    # Log des artefacts
    log_confusion_matrix(y_test, y_pred)
    log_classification_report(y_test, y_pred)
