# module-ia

Module d'intelligence artificielle d'une plateforme développée en équipe.
Il regroupe deux services indépendants, écrits en Python :

1. **Service voix** (`voice_service/`) : traitement de l'audio.
2. **Service compréhension** (`comprehension_service/`) : extraction et structuration de l'information à partir du texte.

> Statut : structure initiale uniquement. Aucune fonctionnalité n'est encore implémentée.

## Les deux services

### 1. Service voix

- Découpage des silences dans les fichiers audio avec **FFmpeg** et **Silero VAD**.
- Transcription audio → texte :
  - **faster-whisper** pour le français ;
  - **MMS-1b** pour le fon.
- Synthèse vocale texte → audio avec **MMS-TTS** et/ou **Piper**.

Ce service est le plus gourmand en ressources : il est isolé dans son propre conteneur **Docker**.

### 2. Service compréhension

- Rapprochement des transcriptions avec un dictionnaire à l'aide de **RapidFuzz**.
- Identification des champs manquants.
- Appel au **LLM** uniquement lorsqu'un champ est manquant.
- Structuration et validation des réponses avec **Instructor** et **Pydantic**.

## Flux prévu

```
Audio ──► [Service voix]
            1. Découpage des silences (FFmpeg + Silero VAD)
            2. Transcription (faster-whisper / MMS-1b)
                     │
                     ▼ texte
          [Service compréhension]
            3. Rapprochement dictionnaire (RapidFuzz)
            4. Détection des champs manquants
            5. Appel LLM si nécessaire (Instructor + Pydantic)
                     │
                     ▼ données structurées et validées
          Plateforme ──► (réponse texte) ──► [Service voix] 6. Synthèse vocale (MMS-TTS / Piper)
```

## Architecture

```
module-ia/
├── README.md
├── .gitignore
├── .env.example                 # Modèle de configuration (sans vraie clé)
├── requirements.txt             # Dépendances communes / outils de dev
│
├── voice_service/               # Service voix (conteneur Docker dédié)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py              # Point d'entrée / API
│       ├── silence_detection.py # FFmpeg + Silero VAD
│       ├── transcription.py     # faster-whisper (fr) + MMS-1b (fon)
│       └── speech_synthesis.py  # MMS-TTS / Piper
│
├── comprehension_service/       # Service compréhension
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py              # Point d'entrée / API
│       ├── dictionary_matching.py # RapidFuzz + champs manquants
│       ├── llm_service.py       # Appel LLM via Instructor
│       └── schemas.py           # Modèles Pydantic
│
├── tests/
│   ├── voice/                   # Tests du service voix
│   └── comprehension/           # Tests du service compréhension
│
├── examples/                    # Exemples d'utilisation et données d'exemple
└── docs/                        # Documentation technique
```

Les deux services sont séparés : chacun possède son propre code (`app/`) et ses propres dépendances (`requirements.txt`), afin de pouvoir être déployés et mis à l'échelle indépendamment.

## Configuration

Copier `.env.example` en `.env` puis renseigner les valeurs. Le fichier `.env` n'est jamais versionné.

## Prochaines étapes

L'installation, le lancement et l'implémentation des services seront documentés au fil des étapes de développement.
