# Image Python officielle
FROM python:3.10-slim

# Dossier de travail
WORKDIR /app

# Copier les fichiers
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l'application FastAPI
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
