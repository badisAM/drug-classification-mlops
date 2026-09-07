"""
Script d'entraînement avec monitoring MLflow + Elasticsearch
Adapté pour votre projet Drug Classification
"""
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from elastic_logger import ElasticsearchLogger
from model_pipeline_mlflow import prepare_data
from sklearn.linear_model import LogisticRegression


def train_model_with_monitoring(C=1.0, max_iter=1000, solver='lbfgs'):
    """
    Entraîne un modèle avec monitoring complet
    """
    
    # Initialiser Elasticsearch Logger
    es_logger = ElasticsearchLogger(
        hosts=['http://localhost:9200'],
        index_name='mlflow-metrics'
    )
    
    # Configurer MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Drug_Classification_Monitoring")
    
    # Charger les données
    print("📊 Chargement des données...")
    X_train, X_test, y_train, y_test, label_encoder = prepare_data("data/drug200.csv")
    
    # Démarrer l'exécution MLflow
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🚀 Démarrage de l'exécution MLflow: {run_id}")
        
        # Log événement de démarrage
        es_logger.log_training_event(
            run_id=run_id,
            event_type='training_start',
            message='Début de l\'entraînement du modèle Drug Classification',
            additional_data={
                'dataset': 'drug200',
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        )
        
        # Log des paramètres
        params = {
            'C': C,
            'max_iter': max_iter,
            'solver': solver,
            'random_state': 42
        }
        mlflow.log_params(params)
        print(f"📝 Paramètres: {params}")
        
        # Entraînement du modèle
        print("🔧 Entraînement du modèle...")
        start_time = time.time()
        
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"⏱️ Temps d'entraînement: {training_time:.2f}s")
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'training_time': training_time
        }
        
        # Log des métriques dans MLflow
        mlflow.log_metrics(metrics)
        
        # Log des métriques dans Elasticsearch
        es_logger.log_metrics(
            run_id=run_id,
            metrics=metrics,
            params=params,
            tags={'experiment': 'Drug_Classification_Monitoring', 'model_type': 'LogisticRegression'}
        )
        
        # Affichage des résultats
        print("\n📈 Métriques du modèle:")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
        
        # Sauvegarder le modèle
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="drug-classifier-monitored"
        )
        
        # Log des informations du modèle dans Elasticsearch
        es_logger.log_model_info(
            run_id=run_id,
            model_name='drug-classifier-monitored',
            model_version='1.0',
            metrics=metrics,
            additional_info={
                'feature_count': X_train.shape[1],
                'solver': solver
            }
        )
        
        # Log événement de fin
        es_logger.log_training_event(
            run_id=run_id,
            event_type='training_end',
            message='Entraînement terminé avec succès',
            additional_data={
                'final_accuracy': metrics['accuracy'],
                'total_time': training_time
            }
        )
        
        print(f"\n✅ Entraînement terminé! Run ID: {run_id}")
        print(f"🔗 MLflow UI: http://localhost:5000")
        print(f"🔗 Kibana: http://localhost:5601")
    
    es_logger.close()
    return run_id, metrics


def train_multiple_experiments():
    """Entraîne plusieurs modèles avec différents hyperparamètres"""
    
    configurations = [
        {'C': 0.1, 'max_iter': 1000, 'solver': 'lbfgs'},
        {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
        {'C': 10.0, 'max_iter': 1000, 'solver': 'lbfgs'},
        {'C': 1.0, 'max_iter': 2000, 'solver': 'lbfgs'},
    ]
    
    print("🔄 Lancement de plusieurs expériences...\n")
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n{'='*50}")
        print(f"Expérience {i}/{len(configurations)}")
        print(f"{'='*50}")
        
        run_id, metrics = train_model_with_monitoring(**config)
        results.append({
            'run_id': run_id,
            'config': config,
            'metrics': metrics
        })
        
        time.sleep(2)  # Pause entre les exécutions
    
    # Afficher le résumé
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DES EXPÉRIENCES")
    print("="*50)
    
    for i, result in enumerate(results, 1):
        print(f"\nExpérience {i}:")
        print(f"  Config: {result['config']}")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {result['metrics']['f1_score']:.4f}")
    
    # Meilleur modèle
    best_result = max(results, key=lambda x: x['metrics']['accuracy'])
    print("\n🏆 Meilleur modèle:")
    print(f"  Config: {best_result['config']}")
    print(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"  Run ID: {best_result['run_id']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "multiple":
        train_multiple_experiments()
    else:
        train_model_with_monitoring()
