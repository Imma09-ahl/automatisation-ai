# module-ia

Composant d'intelligence artificielle d'une plateforme développée en équipe.

Ce module est développé et maintenu de façon indépendante. Il expose deux services distincts, chacun ayant une responsabilité bien délimitée.

---

## Architecture générale

```
module-ia/
├── voice_service/          → Service Voix (conteneurisé, gourmand en ressources)
├── comprehension_service/  → Service Compréhension (logique métier IA)
├── tests/                  → Tests unitaires et d'intégration
├── examples/               → Exemples d'utilisation des services
└── docs/                   → Documentation technique
```

---

## Service 1 — Voice Service

**Rôle :** Traitement audio de bout en bout.

| Fonctionnalité | Outil(s) |
|---|---|
| Découpage des silences | FFmpeg + Silero VAD |
| Transcription français | faster-whisper |
| Transcription fon | MMS-1b (Meta) |
| Synthèse vocale | MMS-TTS et/ou Piper |

Ce service est isolé dans un **conteneur Docker** car il requiert des ressources importantes (GPU optionnel, modèles lourds).

---

## Service 2 — Comprehension Service

**Rôle :** Compréhension et structuration des transcriptions.

| Fonctionnalité | Outil(s) |
|---|---|
| Rapprochement avec dictionnaire | RapidFuzz |
| Identification des champs manquants | Logique interne |
| Appel LLM conditionnel | OpenAI / autre LLM |
| Validation des réponses | Instructor + Pydantic |

Ce service applique une logique d'économie : **le LLM n'est appelé que si un champ manquant ne peut pas être résolu par le dictionnaire**.

---

## Langues supportées

- 🇫🇷 Français
- 🇧🇯 Fon (langue béninoise)

---

## Prérequis

- Python 3.10+
- Docker (pour le service voix)
- FFmpeg installé sur la machine hôte (ou dans le conteneur)

---

## Démarrage rapide

> ⚠️ Cette section sera complétée au fur et à mesure du développement.

```bash
# Copier les variables d'environnement
cp .env.example .env

# Lancer le service voix (Docker)
# À venir

# Lancer le service compréhension
# À venir
```

---

## État du projet

> 🏗️ En cours de construction — Phase 1 : Architecture et structure du projet.
