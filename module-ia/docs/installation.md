# Guide d'Installation — Module IA (`module-ia`)

Ce guide détaille l'installation, la configuration et l'exécution locale des deux microservices du projet `module-ia` : **Voice Service** (Service Voix) et **Comprehension Service** (Service Compréhension).

---

## 1. Prérequis Système

* **Système d'exploitation :** Linux (Ubuntu 20.04+ recommandé), macOS ou Windows 10/11 (64-bit).
* **Python :** Version `3.10` ou `3.11` (Python 3.11.9 recommandé).
* **FFmpeg :** Requis pour le traitement et découpage audio du Service Voix.
  - *Windows :* `winget install Gyan.FFmpeg` ou via le binaire officiel (ajouter `ffmpeg/bin` au PATH).
  - *Linux :* `sudo apt-get install -y ffmpeg`
  - *macOS :* `brew install ffmpeg`
* **Git :** Pour cloner le dépôt.
* **Espace disque conseillé :** 5 à 10 Go disponibles pour les caches de modèles Hugging Face (MMS-1B-All, MMS-TTS).

---

## 2. Clonage du Dépôt et Structure

```bash
git clone https://github.com/votre-organisation/module-ia.git
cd module-ia
```

Le projet est structuré en deux microservices indépendants :

```text
module-ia/
├── voice_service/            # Service voix (Port 8001)
├── comprehension_service/    # Service compréhension (Port 8002)
├── tests/                    # Suites de tests automatisés
├── docs/                     # Documentation technique & contrats
├── models/                   # Cache local des modèles IA (optionnel/volume)
└── outputs/                  # Fichiers audio et transcriptions générés
```

---

## 3. Configuration de l'Environnement Virtuel

Il est recommandé de créer un environnement virtuel unique pour le développement local ou deux environnements isolés par service :

```bash
# Création de l'environnement virtuel Python 3.11
python -m venv .venv

# Activation sous Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activation sous Linux / macOS
source .venv/bin/activate
```

---

## 4. Installation des Dépendances

### A. Dépendances Racine & Outils de Test
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### B. Dépendances du Service Voix
Le service voix dépend de PyTorch CPU (ou CUDA), Silero VAD, Faster-Whisper et Transformers :
```bash
pip install -r voice_service/requirements.txt
```

### C. Dépendances du Service Compréhension
Le service compréhension requiert RapidFuzz, Pydantic et Instructor :
```bash
pip install -r comprehension_service/requirements.txt
```

---

## 5. Configuration des Variables d'Environnement (`.env`)

Copiez le fichier `.env.example` en `.env` :

```bash
cp .env.example .env
```

Éditez le fichier `.env` pour configurer vos clés et paramètres :

```ini
# --- SERVICE VOIX ---
VOICE_DEFAULT_LANGUAGE=fr
VOICE_MODELS_DIR=./models
VOICE_DEVICE=cpu
VOICE_SERVICE_PORT=8001
MAX_AUDIO_SIZE_MB=50

# --- SERVICE COMPRÉHENSION ---
FUZZY_MATCH_THRESHOLD=80.0
FUZZY_PARTIAL_THRESHOLD=85.0
COMPREHENSION_SERVICE_PORT=8002
LLM_ENABLED=true
LLM_API_KEY=votre_cle_api_openai_ici
LLM_MODEL_NAME=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
```

> **Sécurité :** Ne versionnez jamais votre fichier `.env`. Le fichier `.gitignore` est configuré pour l'ignorer systématiquement.

---

## 6. Lancement des Services en Local

Chaque service s'exécute de façon autonome via Uvicorn :

### Lancer le Service Voix (Port 8001)
```powershell
# Depuis la racine du projet
uvicorn voice_service.app.main:app --host 0.0.0.0 --port 8001 --reload
```
Vérification : [http://localhost:8001/health](http://localhost:8001/health)

### Lancer le Service Compréhension (Port 8002)
```powershell
# Dans un second terminal
uvicorn comprehension_service.app.main:app --host 0.0.0.0 --port 8002 --reload
```
Vérification : [http://localhost:8002/health](http://localhost:8002/health)

---

## 7. Exécution de la Suite de Tests

```powershell
# 1. Tests du service voix (transcription, synthèse, silences)
pytest tests/voice -v

# 2. Tests du service compréhension (RapidFuzz, Pydantic, Instructor conditionnel)
pytest tests/comprehension -v

# 3. Tests de bout en bout et interopérabilité des services
pytest tests/test_end_to_end_pipeline.py -v
```
