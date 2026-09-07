# main.py
import argparse
from model_pipeline import (
    prepare_data, train_model, evaluate_model,
    save_model, load_model
)

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Drug Classification (Logistic Regression)")
    parser.add_argument('--step', type=str, choices=['all', 'train', 'evaluate', 'predict'],
                        default='all', help='Étape à exécuter')
    parser.add_argument('--data', type=str, default='data/drug200.csv',
                        help='Chemin vers le fichier CSV')
    
    args = parser.parse_args()

    if args.step in ['all', 'train']:
        # Étape 1 : Préparation des données
        X_train, X_test, y_train, y_test, df = prepare_data(args.data)

        # Étape 2 : Entraînement
        model = train_model(X_train, y_train)

        # Étape 3 : Évaluation
        accuracy = evaluate_model(model, X_test, y_test)

        # Étape 4 : Sauvegarde
        save_model(model)

        print(f"\nPipeline terminé avec succès ! Accuracy = {accuracy:.4f}")

    elif args.step == 'evaluate':
        model = load_model()
        X_train, X_test, y_train, y_test, _ = prepare_data(args.data)
        evaluate_model(model, X_test, y_test)

    elif args.step == 'predict':
        model = load_model()
        print("Modèle chargé. Prédiction prête (à implémenter avec de nouvelles données).")


if __name__ == "__main__":
    main()
    