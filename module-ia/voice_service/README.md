# Voice Service — Détection des silences et découpage audio

Cette première fonctionnalité du service voix reçoit un fichier audio, repère les passages où quelqu'un parle et découpe le fichier en segments WAV, un par passage parlé. Les étapes suivantes (transcription, synthèse vocale) travailleront sur ces segments.

## Rôle de la détection des silences

- Retirer les silences longs, qui n'apportent rien à la transcription.
- Produire des segments courts, plus faciles et plus rapides à transcrire.
- Garder les horodatages (début et fin) de chaque segment dans le fichier original.

## Fonctionnement

```
fichier audio ──► FFmpeg ──► WAV mono 16 kHz PCM 16 bits (temporaire)
                                   │
                                   ▼
                     Silero VAD (get_speech_timestamps)
                                   │  portions parlées (en échantillons)
                                   ▼
                FFmpeg (atrim) ──► segment_0001.wav, segment_0002.wav, ...
                                   │
                                   ▼
                              metadata.json
```

### Silero VAD

Silero VAD est un petit réseau de neurones de détection d'activité vocale (*Voice Activity Detection*). Il découpe l'audio 16 kHz en fenêtres de 32 ms (512 échantillons) et calcule pour chacune une probabilité de parole. `get_speech_timestamps()` regroupe ensuite les fenêtres au-dessus du seuil en portions parlées, en appliquant les durées minimales et la marge configurées.

- Le modèle est chargé avec `load_silero_vad()`, **une seule fois**, au premier traitement (chargement différé), puis réutilisé.
- **Aucun téléchargement de poids** : ils sont inclus dans le paquet `silero-vad` (`silero_vad/data/silero_vad.jit`).
- Le calcul se fait sur CPU, avec PyTorch.

### FFmpeg

FFmpeg sert à :

1. convertir n'importe quel format qu'il sait lire en **WAV mono, 16 kHz, PCM 16 bits** (le format attendu par Silero VAD) ;
2. extraire chaque segment **à l'échantillon près**, avec le filtre `atrim`.

Il doit être installé et accessible dans le `PATH`. La variable `FFMPEG_BINARY` permet d'indiquer un autre exécutable. Le fichier original n'est jamais modifié : la conversion est écrite dans un dossier temporaire supprimé à la fin du traitement.

### Formats audio acceptés

Tous les formats que votre installation de FFmpeg sait décoder : WAV, MP3, OGG/Opus, FLAC, M4A/AAC, WebM, WMA, AMR… La piste audio d'un fichier vidéo (MP4, MKV, etc.) est aussi utilisée. Pour lister les formats pris en charge :

```powershell
ffmpeg -formats
ffmpeg -decoders
```

## Paramètres

| Paramètre | Variable d'environnement | Défaut | Effet |
|---|---|---|---|
| `threshold` | `VAD_THRESHOLD` | `0.5` | Seuil de probabilité de parole (strictement entre 0 et 1). Plus haut : moins de faux positifs, mais risque de couper une voix faible. |
| `min_speech_duration_ms` | `VAD_MIN_SPEECH_DURATION_MS` | `250` | Les portions parlées plus courtes sont ignorées (clics, bruits brefs). |
| `min_silence_duration_ms` | `VAD_MIN_SILENCE_DURATION_MS` | `500` | Durée minimale d'un silence pour séparer deux segments. Plus haut : moins de segments, plus longs. |
| `speech_pad_ms` | `VAD_SPEECH_PAD_MS` | `100` | Marge conservée avant et après chaque portion parlée, pour ne pas couper les débuts et fins de mots. |

Ordre de priorité : valeurs passées dans le code ou en ligne de commande, puis variables d'environnement, puis valeurs par défaut. Une valeur invalide lève `InvalidVADConfigError`.

## Utilisation

### En Python

Depuis le dossier `voice_service/` (même convention d'import `app.` que `main.py`) :

```python
from app.silence_detection import split_audio_on_silences, detect_speech_segments
from app.vad_config import VADConfig

# Découpage complet : segments WAV + metadata.json
result = split_audio_on_silences(
    "conversation.mp3",
    output_root="outputs/audio",
    config=VADConfig(min_silence_duration_ms=700),
)
print(result["segments_count"], result["output_dir"])

# Horodatages seulement, sans rien écrire sur le disque
segments = detect_speech_segments("conversation.mp3")  # [{"start": 0.5, "end": 4.2}, ...]
```

### En ligne de commande (PowerShell, depuis `module-ia/`)

```powershell
cd voice_service
..\.venv\Scripts\python.exe -m app.silence_detection ..\examples\audio\demo_parole.wav --output-dir ..\outputs\audio
# Paramètres optionnels : --threshold 0.6 --min-speech-duration-ms 300 --min-silence-duration-ms 700 --speech-pad-ms 50
```

La commande affiche le JSON du résultat. En cas d'erreur, elle affiche un message explicite et renvoie le code 1.

## Structure des résultats

```
outputs/audio/<nom_du_fichier>/
├── segment_0001.wav
├── segment_0002.wav
└── metadata.json
```

Si un dossier existe déjà pour ce nom de fichier, ses anciens `segment_*.wav` et `metadata.json` sont remplacés. Les segments sont en WAV mono, 16 kHz, PCM 16 bits.

Exemple de `metadata.json` (valeurs illustratives) :

```json
{
  "source": "conversation.wav",
  "duration": 15.5,
  "segments_count": 2,
  "sample_rate": 16000,
  "segments": [
    {"index": 1, "start": 0.5, "end": 4.2, "duration": 3.7, "file": "segment_0001.wav"},
    {"index": 2, "start": 7.0, "end": 11.5, "duration": 4.5, "file": "segment_0002.wav"}
  ]
}
```

Les temps sont en secondes, arrondis à la milliseconde, et marge `speech_pad_ms` comprise. `file` est relatif au dossier du `metadata.json`.

## Erreurs

Toutes les erreurs héritent de `VoiceProcessingError` (`app/audio_errors.py`) :

| Exception | Cas |
|---|---|
| `AudioFileNotFoundError` | Fichier inexistant, ou chemin qui n'est pas un fichier |
| `EmptyAudioFileError` | Fichier de 0 octet ou sans aucun échantillon audio |
| `FFmpegNotFoundError` | FFmpeg absent du système / du `PATH` |
| `UnsupportedAudioFormatError` | Format que FFmpeg ne sait pas lire |
| `AudioConversionError` | Autre échec de la conversion FFmpeg |
| `VADDetectionError` | Échec du chargement du modèle ou de la détection |
| `NoSpeechDetectedError` | Aucune parole détectée (aucun dossier de sortie n'est créé) |
| `SegmentExtractionError` | Échec de l'extraction d'un segment, ou segment vide |
| `InvalidVADConfigError` | Paramètre VAD invalide |

## Organisation du code

| Fichier | Responsabilité |
|---|---|
| `app/silence_detection.py` | Orchestration du pipeline, `metadata.json`, ligne de commande |
| `app/audio_ffmpeg.py` | Appels FFmpeg : conversion, lecture du WAV, extraction des segments |
| `app/vad_model.py` | Chargement différé de Silero VAD et appel à `get_speech_timestamps()` |
| `app/vad_config.py` | Paramètres (`VADConfig`) et lecture des variables d'environnement |
| `app/audio_errors.py` | Exceptions explicites |

## Tests

Depuis `module-ia/` :

```powershell
# Tous les tests du service voix
.\.venv\Scripts\python.exe -m pytest tests/voice -v

# Tests unitaires seulement (modèle simulé par des mocks)
.\.venv\Scripts\python.exe -m pytest tests/voice -v -m "not integration"

# Tests d'intégration seulement (vrai FFmpeg + vrai modèle Silero VAD)
.\.venv\Scripts\python.exe -m pytest tests/voice -v -m integration
```

- Les tests unitaires remplacent Silero VAD par des mocks : ils vérifient la logique du module, **pas** la qualité de détection du modèle.
- Les tests d'intégration utilisent le vrai modèle. Le test sur de la vraie parole a besoin d'un fichier de démonstration contenant au moins deux phrases séparées par un silence. Par défaut il s'agit de `examples/audio/demo_parole.wav` ; la variable `VOICE_DEMO_AUDIO` permet d'en indiquer un autre. Sans ce fichier, ce test est ignoré (*skipped*).
