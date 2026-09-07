# Makefile pour automatiser le projet ML Drug Classification (Logistic Regression)
# Utilisation :
#   make all          # Installe les dépendances + exécute le pipeline complet
#   make install      # Installe les dépendances
#   make data         # Crée les dossiers data/ et models/
#   make train        # Entraîne et évalue (équiv. --step all)
#   make evaluate     # Évalue le modèle sauvegardé
#   make predict      # Charge le modèle pour prédictions
#   make clean        # Nettoie les modèles
#   make requirements # Génère requirements.txt
#   make lint         # Formate le code (optionnel, nécessite black)

.PHONY: all install data train evaluate predict clean requirements lint

# Chemins
DATA_DIR := data
MODELS_DIR := models
MODEL_PATH := $(MODELS_DIR)/logistic_regression_model.joblib
DATA_FILE := $(DATA_DIR)/drug200.csv
MAIN_SCRIPT := main.py
PIPELINE_SCRIPT := model_pipeline.py

# Cible par défaut
all: install data requirements train

# Installation des dépendances (sans fichier requirements.txt externe)
install:
	pip install --upgrade pip
	pip install pandas numpy scikit-learn joblib

# Création des dossiers nécessaires
data:
	mkdir -p $(DATA_DIR) $(MODELS_DIR)
	@echo "Dossiers créés : $(DATA_DIR)/ et $(MODELS_DIR)/"
	@echo "⚠️  Copiez manuellement drug200.csv dans $(DATA_FILE)"
	@if [ ! -f "$(DATA_FILE)" ]; then \
		echo "Erreur : $(DATA_FILE) manquant ! Copiez-le depuis vos attachments."; \
		exit 1; \
	fi

# Génération de requirements.txt
requirements:
	pip freeze | grep -E '^(pandas|numpy|scikit-learn|joblib)' > requirements.txt
	@echo "requirements.txt généré."

# Entraînement complet (préparation + train + evaluate + save)
train:
	@make data
	python $(MAIN_SCRIPT) --step all

# Évaluation seule
evaluate:
	@make data
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "Erreur : Modèle non trouvé ($(MODEL_PATH)). Exécutez 'make train' d'abord."; \
		exit 1; \
	fi
	python $(MAIN_SCRIPT) --step evaluate

# Prédiction (charge le modèle)
predict:
	@make data
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "Erreur : Modèle non trouvé ($(MODEL_PATH)). Exécutez 'make train' d'abord."; \
		exit 1; \
	fi
	python $(MAIN_SCRIPT) --step predict

# Nettoyage
clean:
	rm -rf $(MODELS_DIR)/
	rm -f requirements.txt
	@echo "Nettoyage terminé."

# Formatage du code (optionnel)
lint:
	@pip install black || true
	black $(MAIN_SCRIPT) $(PIPELINE_SCRIPT)
	@echo "Code formaté avec black."



mlflow:
	mlflow server \
	  --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns \
	  --host 0.0.0.0 \
	  --port 5000

run:
	python run_mlflow.py
