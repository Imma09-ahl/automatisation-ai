# ============================================================
# Dockerfile — Voice Service (module-ia)
# ============================================================
# Ce conteneur isole le service voix du reste de la plateforme.
# Il embarque FFmpeg, PyTorch CPU et les moteurs ASR / TTS.
# Les poids des modèles sont persistés via volume monté (/models).
# ============================================================

FROM python:3.11-slim

LABEL maintainer="module-ia"
LABEL description="Service Voix : VAD Silero, Faster-Whisper, MMS-1B-All, MMS-TTS"

WORKDIR /app

# Installation de FFmpeg pour la conversion et le découpage audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Code source
COPY app/ ./app/

# Répertoires de données
RUN mkdir -p /app/outputs/audio/sessions /models/huggingface

EXPOSE 8001

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
