
# hyperparameter_tuning.py
import joblib
import mlflow
from model_pipeline_mlflow import prepare_data
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn

# Préparer les données
X_train, X_test, y_train, y_test, _, _, _ = prepare_data()

# Configurer MLflow
mlflow.set_experiment("Hyperparameter_Tuning")
mlflow.set_tracking_uri("http://localhost:5000")

# Différents hyperparamètres à tester
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
solvers = ['lbfgs', 'liblinear', 'sag', 'saga']
max_iters = [500, 1000, 2000]

print("🎯 Début du tuning d'hyperparamètres avec MLflow...")
print(f"   {len(C_values)} valeurs de C")
print(f"   {len(solvers)} solveurs")
print(f"   {len(max_iters)} valeurs de max_iter")
print("=" * 50)

run_count = 0
best_accuracy = 0
best_params = {}

for C in C_values:
    for solver in solvers:
        for max_iter in max_iters:
            run_count += 1
            
            with mlflow.start_run(run_name=f"run_{run_count}"):
                # Paramètres
                params = {
                    "C": C,
                    "solver": solver,
                    "max_iter": max_iter,
                    "random_state": 42
                }
                
                mlflow.log_params(params)
                
                # Entraînement
                try:
                    model = LogisticRegression(**params)
                    model.fit(X_train, y_train)
                    
                    # Évaluation
                    accuracy = model.score(X_test, y_test)
                    mlflow.log_metric("accuracy", accuracy)
                    
                    # Mettre à jour le meilleur modèle
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = params.copy()
                    
                    print(f"Run {run_count}: C={C}, solver={solver}, "
                          f"max_iter={max_iter}, accuracy={accuracy:.4f}")
                    
                except Exception as e:
                    print(f"❌ Erreur avec {params}: {str(e)}")
                    mlflow.log_param("error", str(e))

print("\n" + "=" * 50)
print("🏆 MEILLEUR MODÈLE TROUVÉ:")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   Paramètres: {best_params}")
print(f"   Total des runs: {run_count}")
print("=" * 50)

# Sauvegarder le meilleur modèle
if best_params:
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_accuracy)
        
        best_model = LogisticRegression(**best_params)
        best_model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(best_model, "best_logistic_regression")
        joblib.dump(best_model, "models/best_model_mlflow.joblib")
        
        print("✅ Meilleur modèle sauvegardé dans MLflow!")
