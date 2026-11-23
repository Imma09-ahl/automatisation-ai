# ============================================================
# Dockerfile — Voice Service
# ============================================================
# Ce conteneur isole le service voix du reste de la plateforme.
# Il embarque FFmpeg, les dépendances Python et les modèles IA.
# ============================================================

# Image de base Python (slim pour limiter la taille)
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="module-ia"
LABEL description="Service Voix : silence detection, transcription, speech synthesis"

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installation de FFmpeg (requis pour le découpage audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source du service
COPY app/ ./app/

# Port exposé par le service voix
EXPOSE 8001

# Commande de démarrage
# À compléter lors de l'implémentation
CMD ["python", "-m", "app.main"]
