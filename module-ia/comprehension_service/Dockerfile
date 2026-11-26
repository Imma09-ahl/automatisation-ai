# ============================================================
# Dockerfile — Comprehension Service (module-ia)
# ============================================================
# Conteneur ultra-léger et haute vélocité.
# Embarque uniquement RapidFuzz, Pydantic, Instructor et FastAPI.
# Ne contient ni PyTorch, ni FFmpeg.
# ============================================================

FROM python:3.11-slim

LABEL maintainer="module-ia"
LABEL description="Service Compréhension : RapidFuzz, Pydantic, Instructor/LLM"

WORKDIR /app

# Dépendances Python isolées
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Code source et dictionnaires
COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8002

ENV PYTHONUNBUFFERED=1
ENV DICTIONARIES_DIR=/app/data/dictionaries

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
