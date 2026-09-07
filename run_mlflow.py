# run_mlflow.py
"""
Script principal pour exécuter, comparer et charger des expériences MLflow
pour le projet Drug Classification.
"""

import argparse
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

from model_pipeline_mlflow import (
    run_mlflow_experiment,
    load_model_from_mlflow,
    prepare_data
)

EXPERIMENT_NAME = "Drug_Classification_Experiment"


# ============================================================================
# 1️⃣ EXPÉRIENCE UNIQUE
# ============================================================================
def run_single_experiment(max_iter: int, C: float, solver: str):
    run_name = f"LR_C{C}_iter{max_iter}_{solver}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n🎯 EXPÉRIENCE UNIQUE")
    print(f"➡ Paramètres: max_iter={max_iter}, C={C}, solver={solver}")

    model, accuracy = run_mlflow_experiment(
        data_path="data/drug200.csv",
        max_iter=max_iter,
        C=C,
        solver=solver,
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name
    )

    print(f"✅ Accuracy obtenue: {accuracy:.4f}")
    return model, accuracy


# ============================================================================
# 2️⃣ EXPÉRIENCES MULTIPLES (GRID SEARCH SIMPLE)
# ============================================================================
def run_multiple_experiments():
    print("\n🔬 EXPÉRIENCES MULTIPLES (GRID)")

    configs = [
        {"max_iter": 1000, "C": 0.1, "solver": "lbfgs"},
        {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
        {"max_iter": 1000, "C": 10.0, "solver": "lbfgs"},
        {"max_iter": 2000, "C": 1.0, "solver": "lbfgs"},
        {"max_iter": 1000, "C": 1.0, "solver": "liblinear"},
        {"max_iter": 1000, "C": 0.5, "solver": "saga"},
    ]

    results = []

    for i, cfg in enumerate(configs, start=1):
        print(f"\n{'='*60}")
        print(f"Expérience {i}/{len(configs)} → {cfg}")
        print(f"{'='*60}")

        try:
            model, acc = run_single_experiment(**cfg)
            results.append({**cfg, "accuracy": acc})
        except Exception as e:
            print(f"❌ Erreur: {e}")

    # Résumé
    print("\n📊 RÉSUMÉ GLOBAL")
    for r in results:
        print(f"{r} → Accuracy={r['accuracy']:.4f}")

    # Meilleur modèle
    best = max(results, key=lambda x: x["accuracy"])
    print("\n🏆 MEILLEURE CONFIGURATION")
    print(best)

    return best


# ============================================================================
# 3️⃣ CHARGER & TESTER UN MODÈLE DEPUIS MLFLOW
# ============================================================================
def load_and_test_model(run_id: str):
    print(f"\n🔄 Chargement du modèle MLflow (run_id={run_id})")

    model = load_model_from_mlflow(run_id)

    X_train, X_test, y_train, y_test, _ = prepare_data("data/drug200.csv")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n✅ ÉVALUATION DU MODÈLE CHARGÉ")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model


# ============================================================================
# 4️⃣ CLI PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline MLflow – Drug Classification")

    parser.add_argument(
        "--mode",
        choices=["single", "multiple", "load"],
        default="single",
        help="Mode d'exécution"
    )

    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear", "saga", "sag", "newton-cg"]
    )

    parser.add_argument("--run_id", type=str, help="Run ID MLflow (mode load)")

    args = parser.parse_args()

    if args.mode == "single":
        run_single_experiment(args.max_iter, args.C, args.solver)

    elif args.mode == "multiple":
        run_multiple_experiments()

    elif args.mode == "load":
        if not args.run_id:
            raise ValueError("❌ --run_id est obligatoire en mode load")
        load_and_test_model(args.run_id)

    print("\n✅ PIPELINE TERMINÉ")
    print("🌐 MLflow UI → http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
