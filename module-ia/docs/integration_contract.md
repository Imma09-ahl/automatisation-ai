# Contrat d'Intégration — Module IA (`module-ia`)

Ce document constitue la référence technique et architecturale pour l'intégration du `module-ia` dans la plateforme collective développée par l'équipe.
Le projet `module-ia` conserve son propre dépôt GitHub indépendant et expose des interfaces claires, modulaires et standardisées.

---

## 1. Architecture Générale & Rôle des Services

Le projet est découpé en deux microservices strictement indépendants et découplés :

```
                          ┌──────────────────────────────────────┐
                          │     Plateforme Métier Principale     │
                          │        (Backend de l'équipe)         │
                          └──────────┬────────────────┬──────────┘
                                     │                │
                      Requêtes Audio │                │ Requêtes JSON
                      (WAV/MP3/...)  │                │ (Texte transcrit)
                                     ▼                ▼
         ┌──────────────────────────────┐          ┌──────────────────────────────┐
         │     SERVICE VOIX (Port 8001) │          │ SERVICE COMPRÉHENSION (8002) │
         │   (GPU/CPU intensif, FFmpeg) │          │    (Léger, Haute vélocité)   │
         ├──────────────────────────────┤          ├──────────────────────────────┤
         │ - Détection silences (VAD)   │          │ - Normalisation Unicode NFC  │
         │ - Transcription FR/FON/YOR   │          │ - Rapprochement RapidFuzz    │
         │ - Synthèse vocale FR/FON/YOR │          │ - Extraction LLM conditionnel│
         └──────────────────────────────┘          └──────────────────────────────┘
```

### A. Service Voix (`voice_service`)
- **Responsabilités :**
  1. Détection des silences et découpage de flux audio en segments de parole (`Silero VAD` + `FFmpeg`).
  2. Transcription vocale multilingue (`faster-whisper` pour le français, `facebook/mms-1b-all` avec adaptateurs pour le fon et le yoruba).
  3. Synthèse vocale text-to-speech multilingue (`facebook/mms-tts-fra`, `facebook/mms-tts-fon`, `facebook/mms-tts-yor`).
- **Ressources :** Gourmand en mémoire RAM/VRAM et calcul (PyTorch, bibliothèques C++ de transcription).

### B. Service Compréhension (`comprehension_service`)
- **Responsabilités :**
  1. Prétraitement linguistique non-destructif (préservation des diacritiques fon et yoruba).
  2. Rapprochement lexical immédiat contre un dictionnaire métier via distance d'édition (`RapidFuzz`).
  3. Détection des intentions et entités avec seuils de confiance configurables.
  4. Décision conditionnelle : déclenchement du LLM (`Instructor` + modèle cible) **uniquement** si des champs indispensables sont manquants.
  5. Validation et formatage standardisé (`Pydantic`).
- **Ressources :** Ultra-léger, très faible latence (matching en mémoire de quelques millisecondes).

---

## 2. Standardisation des Données & Codes Linguistiques

Pour garantir l'interopérabilité entre les deux services et la plateforme collective, les codes linguistiques sont standardisés selon la norme ISO 639-3 canonique :

| Langue | Code Canonique Officiel | Variantes Acceptées | Moteur Transcription | Moteur Synthèse (TTS) | Dictionnaire Compréhension |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Français** | `fr` | `fra` | `faster-whisper` (base, int8) | `facebook/mms-tts-fra` | `data/dictionaries/fr.json` |
| **Fon** | `fon` | `fon` | `mms-1b-all` (adapter: fon) | `facebook/mms-tts-fon` | `data/dictionaries/fon.json` |
| **Yoruba** | `yor` | `yo` | `mms-1b-all` (adapter: yor) | `facebook/mms-tts-yor` | `data/dictionaries/yor.json` |

### Préservation Unicode Impérative
Les échanges JSON doivent impérativement être encodés en `UTF-8` sans échappement destructeur (`ensure_ascii=False`).
Les caractères phonétiques et diacritiques suivants doivent voyager intacts :
- **Fon :** `ɛ`, `ɔ`, `ɖ`, voyelles nasalisées, tons aigu (`á`), grave (`à`), circonflexe (`â`), caron (`ǎ`).
- **Yoruba :** `ẹ`, `ọ`, `ṣ`, voyelles avec tons haut, bas et moyen.

---

## 3. Spécification Détaillée des Endpoints (Contrat REST)

### 3.1 Service Voix (`http://<host>:8001`)

#### 1. Vérification de disponibilité
- **URL :** `GET /health`
- **Réponse HTTP 200 :**
  ```json
  {
    "status": "ok",
    "service": "voice",
    "version": "0.1.0"
  }
  ```

#### 2. Détection des silences et segmentation audio
- **URL :** `POST /vad/split`
- **Méthode :** `POST (multipart/form-data)`
- **Paramètres :**
  - `file` (UploadFile, requis) : Fichier audio (WAV, MP3, OGG, M4A).
  - `threshold` (float, optionnel, défaut : 0.5) : Seuil de détection de parole Silero.
- **Réponse HTTP 200 :**
  ```json
  {
    "source_file": "audio_recu.wav",
    "total_duration": 14.52,
    "speech_segments_count": 3,
    "segments": [
      {
        "index": 1,
        "start": 0.48,
        "end": 3.82,
        "duration": 3.34,
        "audio_file": "segment_001.wav"
      },
      {
        "index": 2,
        "start": 4.12,
        "end": 8.05,
        "duration": 3.93,
        "audio_file": "segment_002.wav"
      }
    ]
  }
  ```
- **Codes d'erreur :** `400 Bad Request` (fichier non audio), `422 Unprocessable Entity` (audio sans parole ou corrompu), `500 Internal Error`.

#### 3. Récupération sécurisée d'un segment audio découpé
- **URL :** `GET /audio/segments/{session_id}/{filename}`
- **Méthode :** `GET`
- **Paramètres d'URL :**
  - `session_id` : Identifiant de session hexadécimal généré lors du split.
  - `filename` : Nom du fichier segment (ex: `segment_0001.wav`).
- **Sécurité :** Protection intégrée contre la traversée de chemin (*Path Traversal*).
- **Réponse HTTP 200 :** Fichier binaire audio (`Content-Type: audio/wav`).
- **Codes d'erreur :** `403 Forbidden` (tentative de traversée de répertoire), `404 Not Found` (segment introuvable).

#### 4. Transcription vocale multilingue
- **URL :** `POST /transcribe`
- **Méthode :** `POST (multipart/form-data)`
- **Paramètres :**
  - `file` (UploadFile, requis) : Fichier audio.
  - `language` (string, requis) : `"fr"`, `"fon"` ou `"yor"`.
- **Réponse HTTP 200 :**
  ```json
  {
    "text": "Je veux envoyer de l'argent à maman en fcfa",
    "language": "fr",
    "duration": 4.25,
    "segments": [
      {
        "id": 1,
        "start": 0.0,
        "end": 4.25,
        "text": "Je veux envoyer de l'argent à maman en fcfa"
      }
    ]
  }
  ```
- **Codes d'erreur :** `400 Bad Request` (langue non supportée), `500 Internal Error`.

#### 4. Synthèse vocale multilingue (Text-to-Speech)
- **URL :** `POST /synthesize`
- **Méthode :** `POST (application/json)`
- **Corps de la requête :**
  ```json
  {
    "text": "Akwɛ́ sɛ́dó nɔ̂ en fcfa",
    "language": "fon",
    "engine": "mms-tts"
  }
  ```
- **Réponse HTTP 200 :** Fichier binaire audio `audio/wav` téléchargeable ou streamable (`Content-Type: audio/wav`).
- **Codes d'erreur :** `400 Bad Request` (langue inconnue ou texte vide), `500 Internal Error`.

---

### 3.2 Service Compréhension (`http://<host>:8002`)

#### 1. Vérification de disponibilité
- **URL :** `GET /health`
- **Réponse HTTP 200 :**
  ```json
  {
    "status": "ok",
    "service": "comprehension",
    "supported_languages": ["fr", "fon", "yor"],
    "version": "0.2.0"
  }
  ```

#### 2. Analyse sémantique et extraction structurée
- **URL :** `POST /comprehension/analyze` (alias: `/understand`)
- **Méthode :** `POST (application/json)`
- **Corps de la requête :**
  ```json
  {
    "text": "Je veux envoyer de l'argent à Kofi en fcfa",
    "language": "fr",
    "expected_fields": ["intent", "destinataire", "devise"],
    "source": "voice_service_transcription"
  }
  ```
- **Réponse HTTP 200 :**
  ```json
  {
    "source": "voice_service_transcription",
    "language": "fr",
    "raw_text": "Je veux envoyer de l'argent à Kofi en fcfa",
    "normalized_text": "je veux envoyer de l'argent à kofi en fcfa",
    "intent": {
      "field_name": "intent",
      "value": "transfert_argent",
      "canonical_value": "transfert_argent",
      "score": 100.0,
      "source": "dictionary"
    },
    "entities": {
      "destinataire": {
        "field_name": "destinataire",
        "value": "Kofi",
        "canonical_value": "Kofi",
        "score": null,
        "source": "llm"
      },
      "devise": {
        "field_name": "devise",
        "value": "fcfa",
        "canonical_value": "fcfa",
        "score": 100.0,
        "source": "dictionary"
      }
    },
    "available_fields": ["intent", "devise", "destinataire"],
    "missing_fields": [],
    "llm_used": true,
    "is_complete": true
  }
  ```
- **Traçabilité `source` dans chaque champ :**
  - `"dictionary"` : Résolu par RapidFuzz (score de 0 à 100).
  - `"llm"` : Complété par Instructor/LLM car absent du dictionnaire.
  - `"missing"` : Information introuvable (valeur `null`).
- **Codes d'erreur :** `422 Unprocessable Entity` (texte vide ou langue absente), `500 Internal Error`.

---

## 4. Modalités de Communication entre la Plateforme Principale et le Module IA

La plateforme principale peut communiquer avec le module IA selon deux modes :

1. **Pipeline découplé (Recommandé) :**
   - L'utilisateur envoie son enregistrement audio au backend principal.
   - Le backend principal appelle `POST voice_service:8001/transcribe` $\rightarrow$ reçoit le texte transcrit et la langue détectée/validée.
   - Le backend principal appelle `POST comprehension_service:8002/comprehension/analyze` avec le texte transcrit $\rightarrow$ reçoit le JSON métier structuré.
   - Si une réponse vocale est nécessaire, le backend principal transmet le texte généré à `POST voice_service:8001/synthesize` $\rightarrow$ reçoit le WAV à rejouer.

2. **Pipeline direct (Orchestration interne au Module IA) :**
   - Un endpoint de façade peut orchestrer l'enchaînement transcription $\rightarrow$ compréhension en une seule requête si la plateforme principale le souhaite.

---

## 5. Architecture de Déploiement Docker & Docker Compose Proposée

Pour garantir une isolation parfaite des dépendances et un dimensionnement adapté des ressources :

### Découpage en 2 Conteneurs
1. **Conteneur `voice_service` :**
   - Image : `python:3.10-slim` ou `python:3.11-slim`.
   - Paquets système : `ffmpeg`.
   - Modèles montés en volume persistant pour éviter tout re-téléchargement (`/models:/app/models`).
   - Expose le port `8001`.
2. **Conteneur `comprehension_service` :**
   - Image ultra-légère `python:3.11-slim` sans PyTorch ni FFmpeg.
   - Dépendances : `rapidfuzz`, `pydantic`, `instructor`, `openai`, `fastapi`, `uvicorn`.
   - Volume monté pour les dictionnaires métier (`/data/dictionaries:/app/data/dictionaries`).
   - Expose le port `8002`.

### Proposition de configuration `docker-compose.yml` (À déployer ultérieurement)
```yaml
version: '3.8'

services:
  voice-service:
    build:
      context: ./voice_service
      dockerfile: Dockerfile
    container_name: module_ia_voice
    ports:
      - "8001:8001"
    environment:
      - WHISPER_MODEL_SIZE=base
      - HF_HOME=/models/huggingface
    volumes:
      - ./models:/models
      - ./outputs:/app/outputs
    restart: unless-stopped
    networks:
      - module_ia_net

  comprehension-service:
    build:
      context: ./comprehension_service
      dockerfile: Dockerfile
    container_name: module_ia_comprehension
    ports:
      - "8002:8002"
    environment:
      - COMPREHENSION_FUZZY_MATCH_THRESHOLD=80.0
      - LLM_ENABLED=true
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_MODEL_NAME=gpt-4o-mini
    volumes:
      - ./comprehension_service/data/dictionaries:/app/data/dictionaries
    restart: unless-stopped
    networks:
      - module_ia_net

networks:
  module_ia_net:
    driver: bridge
```

---

## 6. Variables d'Environnement Requises

### Pour le Service Voix
- `WHISPER_MODEL_SIZE` : Modèle whisper (`base`, `small`, etc. ; défaut : `base`).
- `WHISPER_DEVICE` : Périphérique de calcul (`cpu` ou `cuda` ; défaut : `cpu`).
- `WHISPER_COMPUTE_TYPE` : Précision (`int8` ou `float32` ; défaut : `int8`).
- `HF_HOME` : Répertoire de cache pour les poids MMS-1B-All et MMS-TTS.

### Pour le Service Compréhension
- `COMPREHENSION_FUZZY_MATCH_THRESHOLD` : Seuil de correspondance RapidFuzz (0 à 100, recommandé : `80.0`).
- `DICTIONARIES_DIR` : Chemin des dictionnaires JSON par langue.
- `LLM_ENABLED` : Activer/désactiver l'appel conditionnel (`true`/`false`).
- `LLM_API_KEY` : Clé secrète pour le fournisseur LLM.
- `LLM_BASE_URL` : URL de l'API LLM (défaut : `https://api.openai.com/v1`).
- `LLM_MODEL_NAME` : Nom du modèle (défaut : `gpt-4o-mini`).
- `LLM_TIMEOUT_SECONDS` : Timeout maximal d'un appel (défaut : `15.0`).

---

## 7. État des Lieux : Développé vs Reste à Développer

| Composant | Sous-fonctionnalité | Statut Actuel |
| :--- | :--- | :---: |
| **Voix** | Détection des silences Silero VAD + FFmpeg | **100 % Implémenté & Validé** |
| **Voix** | Transcription FR (`faster-whisper`) | **100 % Implémenté & Validé** |
| **Voix** | Transcription Fon & Yoruba (`mms-1b-all`) | **100 % Implémenté & Validé** |
| **Voix** | Synthèse vocale FR/FON/YOR (`mms-tts`) | **100 % Implémenté & Validé** |
| **Voix** | Endpoints HTTP `/transcribe`, `/synthesize`, `/health` | **Implémentés dans `main.py`** |
| **Compréhension** | Normalisation Unicode NFC & Diacritiques | **100 % Implémenté & Validé** |
| **Compréhension** | Moteur RapidFuzz (ratio, partial_ratio, extract) | **100 % Implémenté & Validé** |
| **Compréhension** | Modèles Pydantic stricts | **100 % Implémenté & Validé** |
| **Compréhension** | Appel conditionnel Instructor / LLM | **100 % Implémenté & Validé** |
| **Compréhension** | Dictionnaires génériques configurables | **100 % Implémenté & Validé** |
| **Compréhension** | Endpoint HTTP `/comprehension/analyze` | **Implémenté dans `main.py`** |
| **Intégration** | Dockerfile spécifique `comprehension_service` | *À formaliser lors de l'intégration* |
| **Intégration** | Fichier `docker-compose.yml` racine | *À formaliser lors de l'intégration* |
| **Métier** | Dictionnaire métier réel de l'entreprise | *En attente des données de l'équipe* |

---

## 8. Questions Techniques à Poser à l'Équipe Plateforme

Avant de lancer le déploiement ou l'interconnexion réseau, voici les 6 points clés à clarifier avec le chef de projet et l'équipe backend :

1. **Langage et architecture de la plateforme hôte :**
   - Quel est le langage principal de la plateforme (Node.js/TypeScript, Java/Spring, PHP/Laravel, Go, Python/Django) ?
   - Les services communiqueront-ils par HTTP REST classique ou un bus de messages (RabbitMQ, Kafka, Redis) est-il envisagé pour l'audio ?
2. **Format des fichiers audio émis :**
   - Quels formats et codecs audio le client web/mobile de l'équipe enverra-t-il exactement (WAV PCM, MP3, WebM/Opus, AAC) ?
   - Les enregistrements seront-ils envoyés en streaming continu (chunks WebSockets) ou en fichier complet après relâchement du bouton de parole ?
3. **Mécanisme d'authentification inter-services :**
   - Quel standard de sécurité sécurisera les appels aux endpoints du module IA (API Key dans un header `X-API-Key`, Bearer Token JWT, réseau privé interne / mTLS) ?
4. **Fournisseur et hébergement du LLM :**
   - Quel fournisseur de LLM sera retenu en production pour la complétion des champs manquants (OpenAI public, Azure OpenAI, modèle local hébergé via vLLM ou Ollama pour des raisons de souveraineté des données) ?
5. **Dictionnaire métier réel :**
   - Qui fournit et maintient le dictionnaire des termes réels du domaine (liste exhaustive des intentions, entités, produits, devises, villes béninoises/nigérianes) ?
   - Sous quel format la plateforme principale souhaite-t-elle mettre à jour ce dictionnaire (fichiers JSON statiques, table SQL, endpoint d'administration) ?
6. **Infrastructure et contraintes de déploiement :**
   - Les conteneurs seront-ils déployés sur un serveur dédié unique avec Docker Compose, ou sur un cluster Kubernetes (K8s) ?
   - Un GPU Nvidia est-il disponible pour accélérer la transcription et la synthèse vocale, ou le déploiement doit-il rester optimisé sur CPU ?
