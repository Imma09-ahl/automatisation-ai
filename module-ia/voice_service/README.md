# Voice Service — Détection des silences, découpage audio et transcription française

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
| `InvalidWhisperConfigError` | Paramètre Whisper invalide |
| `WhisperModelLoadError` | Modèle faster-whisper impossible à charger ou à télécharger |
| `MetadataError` | `metadata.json` absent, illisible ou mal formé, ou fichier de segment manquant |
| `TranscriptionError` | Échec de la transcription d'un segment (le nom du segment est indiqué) |
| `EmptyTranscriptionError` | Aucun texte reconnu dans aucun segment |

## Organisation du code

| Fichier | Responsabilité |
|---|---|
| `app/silence_detection.py` | Orchestration du découpage audio, `metadata.json`, ligne de commande |
| `app/audio_ffmpeg.py` | Appels FFmpeg : conversion, lecture du WAV, extraction des segments |
| `app/vad_model.py` | Chargement différé de Silero VAD et appel à `get_speech_timestamps()` |
| `app/vad_config.py` | Paramètres (`VADConfig`) et lecture des variables d'environnement |
| `app/audio_errors.py` | Exceptions explicites pour l'audio et la transcription |
| `app/transcription.py` | Orchestration trilingue (FR, Fon, Yoruba), calcul d'horodatages globaux et `transcription.json` |
| `app/faster_whisper_engine.py` | Moteur faster-whisper pour le français (chargement différé, modèle `base`) |
| `app/mms_engine.py` | Moteur Meta MMS-1b-all pour le Fon (`fon`) et le Yoruba (`yor`) |
| `app/transcription_fr.py` | Module spécifique français historique |
| `app/whisper_model.py` | Accès au modèle faster-whisper |
| `app/whisper_config.py` | Paramètres (`WhisperConfig`) et lecture des variables `WHISPER_*` |

## Transcription multilingue (Français, Fon, Yoruba)

Le service voix prend en charge **trois langues** via deux moteurs spécialisés :

1. **Français (`fr`)** : `faster-whisper` (OpenAI Whisper optimisé CTranslate2).
2. **Fon (`fon`)** : `facebook/mms-1b-all` (Meta MMS fine-tuné ASR avec adaptateur `fon`).
3. **Yoruba (`yo` / `yor`)** : `facebook/mms-1b-all` (Meta MMS fine-tuné ASR avec adaptateur `yor`).

### 1. Français (faster-whisper)

- **Modèle** : `WhisperModel` (`base` par défaut, configurable : `tiny`, `small`, etc.).
- **Tâche** : `transcribe`, langue : `fr`.
- **Exécution** : CPU avec précision `int8`.
- **Licence** : MIT (usage commercial et non-commercial autorisé).

### 2. Fon et Yoruba (Meta MMS-1b-all)

- **Modèle de base** : `facebook/mms-1b-all` (environ 1 milliard de paramètres).
- **Architecture** : `Wav2Vec2ForCTC` + `AutoProcessor`.
- **Adaptateurs linguistiques** :
  - Fon : `processor.tokenizer.set_target_lang("fon")` et `model.load_adapter("fon")`.
  - Yoruba : `processor.tokenizer.set_target_lang("yor")` et `model.load_adapter("yor")`.
- **Changement dynamique** : Le modèle de base reste chargé en mémoire ; seul l'adaptateur bascule dynamiquement si la langue change entre deux requêtes.
- **Inférence** : CPU sous `torch.no_grad()`.
- **Avertissement Licence** : Le checkpoint `facebook/mms-1b-all` est distribué sous licence **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. Il est strictement réservé à la recherche et à un usage non-commercial.
- **Prérequis matériels MMS** :
  - Disque : ~4 Go à 4,5 Go nécessaires pour le modèle et ses adaptateurs. Le lecteur `D:` (286 Go libres) doit être privilégié face au lecteur `C:` (2 Go libres).
  - RAM : environ 4 Go de mémoire vive pour charger le modèle en FP32 sur CPU.

### Utilisation de la transcription

```python
from app.transcription import transcribe, transcribe_segments_dir

# 1. Découpage automatique puis transcription dans la langue voulue
result_fr = transcribe("conversation.wav", language="fr")
result_fon = transcribe("audio_fon.wav", language="fon")
result_yor = transcribe("audio_yor.wav", language="yor")

# 2. Transcription d'un dossier de segments Silero VAD existant
result = transcribe_segments_dir("outputs/audio/demo_parole", language="fr")
```

En ligne de commande :

```powershell
# Transcription en français
.\.venv\Scripts\python.exe -m voice_service.app.transcription outputs/audio/demo_parole -l fr

# Transcription en fon
.\.venv\Scripts\python.exe -m voice_service.app.transcription outputs/audio/demo_fon -l fon

# Transcription en yoruba (code 'yo' ou 'yor')
.\.venv\Scripts\python.exe -m voice_service.app.transcription outputs/audio/demo_yor -l yo
```

### Structure du résultat (`transcription.json`)

Le schéma JSON généré est strictement identique pour les trois langues :

```json
{
  "source": "conversation.wav",
  "language": "fr",
  "engine": "faster-whisper",
  "model": "base",
  "text": "Bonjour tout le monde. Comment allez-vous ?",
  "segments_count": 2,
  "segments": [
    {
      "index": 1,
      "start": 1.0,
      "end": 4.5,
      "duration": 3.5,
      "file": "segment_0001.wav",
      "text": "Bonjour tout le monde."
    },
    {
      "index": 2,
      "start": 6.0,
      "end": 10.0,
      "duration": 4.0,
      "file": "segment_0002.wav",
      "text": "Comment allez-vous ?"
    }
  ]
}
```

## Transcription française (faster-whisper)

Cette étape transcrit en français chaque `segment_XXXX.wav` produit par la détection des silences.

- **faster-whisper** est une version optimisée (moteur CTranslate2) du modèle Whisper d'OpenAI.
- La langue est imposée (`fr`) : pas de détection automatique.
- Le filtre de parole interne de faster-whisper est désactivé : les segments ont déjà été découpés par Silero VAD.
- Le modèle est chargé au premier traitement, puis réutilisé. Il n'est rechargé que si le modèle, le matériel, la précision, le nombre de threads ou le dossier de stockage changent.

### Téléchargement du modèle

Au premier chargement, les poids sont téléchargés depuis Hugging Face (`Systran/faster-whisper-<taille>`) et stockés dans `%USERPROFILE%\.cache\huggingface\hub` (environ 464 Mo pour `small`). Les traitements suivants n'ont plus besoin d'internet. Sous Windows, l'avertissement « To support symlinks on Windows… » est sans conséquence ; `HF_HUB_DISABLE_SYMLINKS_WARNING=1` le masque.

### Paramètres

| Paramètre | Variable d'environnement | Défaut | Effet |
|---|---|---|---|
| `model_size` | `WHISPER_MODEL_SIZE` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`… ou chemin d'un modèle CTranslate2. Plus grand : plus précis, plus lent. |
| `device` | `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda` (carte NVIDIA) ou `auto`. |
| `compute_type` | `WHISPER_COMPUTE_TYPE` | `int8` | Précision des calculs ; `int8` est le plus rapide sur processeur. |
| `beam_size` | `WHISPER_BEAM_SIZE` | `5` | Largeur de recherche ; plus grand : un peu plus précis, plus lent. |
| `cpu_threads` | `WHISPER_CPU_THREADS` | `0` | Threads processeur (`0` = automatique). |
| `download_root` | `WHISPER_DOWNLOAD_ROOT` | cache Hugging Face | Dossier de stockage des poids. |
| `local_files_only` | `WHISPER_LOCAL_FILES_ONLY` | `false` | `true` interdit tout téléchargement. |

### Utilisation

```python
from app.transcription_fr import split_and_transcribe_fr, transcribe_segments_dir
from app.whisper_config import WhisperConfig

# Découpage + transcription
result = split_and_transcribe_fr("conversation.mp3", output_root="outputs/audio")
print(result["transcription"]["text"])

# Transcription d'un dossier déjà découpé
result = transcribe_segments_dir("outputs/audio/conversation", WhisperConfig(model_size="base"))
```

```powershell
cd voice_service
..\.venv\Scripts\python.exe -m app.transcription_fr ..\examples\audio\demo_parole.wav --output-dir ..\outputs\audio
..\.venv\Scripts\python.exe -m app.transcription_fr --segments-dir ..\outputs\audio\demo_parole
# Options : --model-size base --device cpu --compute-type int8 --beam-size 5 --local-files-only
```

### Résultats

Dans `outputs/audio/<nom_du_fichier>/` :

- `metadata.json` reçoit un champ `text` par segment et un bloc `transcription` ;
- `transcription.txt` contient le texte complet.

```json
{
  "source": "conversation.wav",
  "segments": [
    {"index": 1, "start": 0.5, "end": 4.2, "duration": 3.7, "file": "segment_0001.wav", "text": "Bonjour à tous."}
  ],
  "transcription": {"language": "fr", "engine": "faster-whisper", "model": "small", "text": "Bonjour à tous."}
}
```

Un segment où rien n'est reconnu garde `"text": ""`. `metadata.json` n'est réécrit qu'une fois tous les segments transcrits : en cas d'erreur, le résultat de la détection des silences reste intact.

## Synthèse vocale (Text-to-Speech / MMS-TTS)

Le service voix intègre la synthèse vocale trilingue (**Français, Fon, Yoruba**) basée sur l'architecture VITS de Meta MMS-TTS :

| Langue | Identifiant du modèle | Code langue | Architecture |
|---|---|---|---|
| **Français** | `facebook/mms-tts-fra` | `fr` ou `fra` | `VitsModel` + `AutoTokenizer` |
| **Fon** | `facebook/mms-tts-fon` | `fon` | `VitsModel` + `AutoTokenizer` |
| **Yoruba** | `facebook/mms-tts-yor` | `yo` ou `yor` | `VitsModel` + `AutoTokenizer` |

### Fonctionnalités principales

1. **Architecture VITS native** : Utilise les classes officielles `transformers.VitsModel` et `transformers.AutoTokenizer`.
2. **Préservation Unicode intégrale** : Respecte scrupuleusement les caractères spécifiques du Fon (`ɛ`, `ɔ`, `ɖ`) et les marques diacritiques/tonales du Fon et du Yoruba (sub-dots `ẹ`, `ọ`, `ṣ`, accents aigus, graves, etc.).
3. **Gestion des textes longs** : Découpage intelligent par phrases sans coupure de mots, synthèse successive et concaténation avec silence paramétrable (`silence_between_sentences_s`, 0.25 s par défaut).
4. **Gestion mémoire optimale** : Ne conserve qu'un seul modèle MMS-TTS chargé en RAM à la fois. Lors du changement de langue, le modèle précédent est immédiatement déchargé et la mémoire libérée (`gc.collect()`).
5. **Sauvegarde standardisée** : Écriture de fichiers WAV mono PCM 16 bits (sans dépendances externes via `wave` + `numpy`) dans `outputs/tts/` accompagnés de leur fichier de métadonnées JSON (`<nom>.json`).
6. **Reproductibilité** : Paramètre optionnel `seed` pour fixer la graine aléatoire PyTorch/NumPy.
7. **Place de Piper** : Piper est prévu comme moteur alternatif optionnel pour le français (`engine="piper"`). Piper ne dispose d'aucune voix pour le Fon ni le Yoruba ; MMS-TTS demeure le moteur unifié prioritaire.

### Ressources et licence

- **Taille disque par modèle** : ~138 Mo (`model.safetensors`), soit ~415 Mo pour l'ensemble des trois langues.
- **Empreinte RAM sur CPU** : ~250 à 350 Mo par modèle actif.
- **Licence** : **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

### Utilisation

En Python :

```python
from app.speech_synthesis import synthesize

# 1. Synthèse en français
res_fr = synthesize("Bonjour, bienvenue dans le service voix.", language="fr")
print(res_fr["file"], res_fr["duration"])

# 2. Synthèse en fon
res_fon = synthesize("Mi dɔ nú we ɖɔ: é nyɔ́ tawun, kúdò azɔ̌!", language="fon")

# 3. Synthèse en yoruba (code 'yo' ou 'yor')
res_yor = synthesize("Ẹ kú àárọ̀ o, báwo ni gbogbo nǹkan?", language="yor")
```

En ligne de commande :

```powershell
# Français
.\.venv\Scripts\python.exe -m voice_service.app.speech_synthesis "Bonjour à tous." -l fr -o outputs/tts/speech_fr.wav

# Fon
.\.venv\Scripts\python.exe -m voice_service.app.speech_synthesis "Afon ganjii a." -l fon -o outputs/tts/speech_fon.wav

# Yoruba
.\.venv\Scripts\python.exe -m voice_service.app.speech_synthesis "E kaaro o." -l yo -o outputs/tts/speech_yor.wav
```

### Format du fichier de métadonnées JSON

Pour chaque génération, un fichier `.json` est produit aux côtés du `.wav` :

```json
{
  "text": "Bonjour, bienvenue dans le service voix.",
  "language": "fr",
  "model": "facebook/mms-tts-fra",
  "sample_rate": 16000,
  "duration": 2.45,
  "file": "outputs\\tts\\speech_fr.wav",
  "created_at": "2026-11-23T20:15:00.123456"
}
```

## Tests

Depuis `module-ia/` :

```powershell
# Tous les tests du service voix
.\.venv\Scripts\python.exe -m pytest tests/voice -v

# Tests unitaires seulement (modèle simulé par des mocks)
.\.venv\Scripts\python.exe -m pytest tests/voice -v -m "not integration"

# Tests d'intégration seulement (vrai FFmpeg + vrais modèles Silero VAD et faster-whisper)
.\.venv\Scripts\python.exe -m pytest tests/voice -v -m integration
```

- La transcription est testée dans `tests/voice/test_transcription_fr.py`. Ses tests unitaires remplacent faster-whisper par un faux modèle : ils vérifient la logique du module, **pas** la qualité de la transcription.
- Son test d'intégration utilise le vrai modèle s'il est déjà dans le cache local (aucun téléchargement pendant les tests) ; sinon il est ignoré (*skipped*).

- Les tests unitaires remplacent Silero VAD par des mocks : ils vérifient la logique du module, **pas** la qualité de détection du modèle.
- Les tests d'intégration utilisent le vrai modèle. Le test sur de la vraie parole a besoin d'un fichier de démonstration contenant au moins deux phrases séparées par un silence. Par défaut il s'agit de `examples/audio/demo_parole.wav` ; la variable `VOICE_DEMO_AUDIO` permet d'en indiquer un autre. Sans ce fichier, ce test est ignoré (*skipped*).
