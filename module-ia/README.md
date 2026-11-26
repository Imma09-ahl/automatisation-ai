# module-ia

Composant d'intelligence artificielle autonome d'une plateforme développée en équipe.

Ce module est développé et maintenu dans son **dépôt indépendant**. Il expose deux microservices distincts, communicant par API HTTP REST standardisées :
1. **Service Voix (`voice_service`)** — Port 8001 (ASR, TTS, VAD).
2. **Service Compréhension (`comprehension_service`)** — Port 8002 (RapidFuzz, Pydantic, Instructor/LLM).

---

## 1. Langues Prises en Charge

Le projet prend en charge nativement trois langues avec préservation intégrale des diacritiques et tons Unicode :
- 🇫🇷 **Français (`fr`)** (alias: `fra`)
- 🇧🇯 **Fon (`fon`)** — Voyelles ouvertes (`ɛ`, `ɔ`), consonne `ɖ`, tons montants et descendants.
- 🇳🇬 **Yoruba (`yor`)** (alias: `yo`) — Voyelles avec point souscrit (`ẹ`, `ọ`), consonne `ṣ`, tons aigus et graves.

---

## 2. Architecture Générale

```text
module-ia/
├── voice_service/            → Service Voix (FFmpeg, Silero VAD, Faster-Whisper, MMS)
├── comprehension_service/    → Service Compréhension (RapidFuzz, Pydantic, Instructor)
├── tests/                    → Suites de tests unitaires, API et bout en bout
├── docs/                     → Documentation technique, intégration et handover
│   ├── integration_contract.md
│   ├── deployment.md
│   ├── installation.md
│   └── handover.md
├── models/                   → Volume persistant pour les poids de modèles
├── outputs/                  → Fichiers audio et transcriptions générés
└── docker-compose.yml        → Orchestration locale des conteneurs
```

---

## 3. Matrice des Fonctionnalités

| Service | Fonctionnalité | Technologie / Moteur | Statut |
| :--- | :--- | :--- | :---: |
| **Voix** | Détection de silences & découpage | FFmpeg + Silero VAD | **Opérationnel** |
| **Voix** | Transcription Français | `faster-whisper` (base, int8) | **Opérationnel** |
| **Voix** | Transcription Fon & Yoruba | `facebook/mms-1b-all` (adaptateurs fon/yor) | **Opérationnel** |
| **Voix** | Synthèse vocale multilingue | `facebook/mms-tts-{fra,fon,yor}` | **Opérationnel** |
| **Voix** | API HTTP REST (Port 8001) | FastAPI (`/health`, `/vad/split`, `/transcribe`, `/synthesize`) | **Opérationnel** |
| **Compréhension** | Normalisation linguistique | Unicode NFC non-destructif | **Opérationnel** |
| **Compréhension** | Rapprochement lexical | RapidFuzz (`fuzz.ratio`, n-grammes, seuil 80.0) | **Opérationnel** |
| **Compréhension** | Validation des données | Pydantic v2 (`ComprehensionOutput`, `FieldMatch`) | **Opérationnel** |
| **Compréhension** | Extraction LLM conditionnelle | Instructor + OpenAI / API compatible | **Opérationnel** |
| **Compréhension** | API HTTP REST (Port 8002) | FastAPI (`/health`, `/comprehension/analyze`, `/dictionaries/{lang}`) | **Opérationnel** |

---

## 4. Démarrage Rapide

### Option A : Avec Docker Compose (Recommandé)
```bash
# Copier et configurer les variables d'environnement
cp .env.example .env

# Lancer les deux microservices
docker compose up --build -d
```
- Service Voix : `http://localhost:8001/docs` (Swagger UI)
- Service Compréhension : `http://localhost:8002/docs` (Swagger UI)

### Option B : En Local avec Python 3.11
```bash
# Créer et activer le venv
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/macOS

# Installer les dépendances
pip install -r requirements.txt
pip install -r voice_service/requirements.txt
pip install -r comprehension_service/requirements.txt

# Lancer chaque service dans un terminal distinct
uvicorn voice_service.app.main:app --port 8001 --reload
uvicorn comprehension_service.app.main:app --port 8002 --reload
```

---

## 5. Exécution des Tests

```bash
# Tests voix
pytest tests/voice -v

# Tests compréhension
pytest tests/comprehension -v

# Tests d'intégration et pipeline de bout en bout
pytest tests/test_end_to_end_pipeline.py -v
```

---

## 6. Documentation Détaillée

Consultez les guides dédiés dans le dossier `docs/` :
- [docs/integration_contract.md](docs/integration_contract.md) : Contrat d'interface technique et formats JSON pour les développeurs.
- [docs/handover.md](docs/handover.md) : Guide de transmission pour le chef de groupe.
- [docs/installation.md](docs/installation.md) : Instructions d'installation pas-à-pas.
- [docs/deployment.md](docs/deployment.md) : Guide de déploiement en conteneurs Docker.
