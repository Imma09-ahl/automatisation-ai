"""
Détection des silences et découpage automatique d'un fichier audio.

Pipeline :
  1. Vérification du fichier (existence, contenu non vide).
  2. Conversion FFmpeg en WAV mono 16 kHz PCM 16 bits (fichier temporaire).
  3. Détection des portions parlées avec Silero VAD (``get_speech_timestamps``).
  4. Extraction de chaque portion en WAV avec FFmpeg.
  5. Écriture de ``metadata.json``.

Résultat : ``<output_root>/<nom_du_fichier>/segment_0001.wav ... + metadata.json``.
Le fichier original n'est jamais modifié.

Utilisation en ligne de commande (depuis ``voice_service/``) :
    python -m app.silence_detection chemin/vers/audio.mp3 --output-dir ../outputs/audio
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

_voice_service_dir = Path(__file__).resolve().parent.parent
if str(_voice_service_dir) not in sys.path:
    sys.path.insert(0, str(_voice_service_dir))

from app.audio_errors import (
    AudioFileNotFoundError,
    EmptyAudioFileError,
    NoSpeechDetectedError,
    VoiceProcessingError,
)
from app.audio_ffmpeg import convert_to_pcm_wav, extract_segment, find_ffmpeg, load_pcm_wav
from app.vad_config import SAMPLE_RATE, VADConfig
from app.vad_model import detect_speech_timestamps

PathLike = Union[str, Path]

DEFAULT_OUTPUT_ROOT = Path("outputs") / "audio"
METADATA_FILENAME = "metadata.json"
SEGMENT_PATTERN = "segment_{index:04d}.wav"


@dataclass(frozen=True)
class SpeechSegment:
    """Portion parlée détectée, exprimée en échantillons à ``sample_rate`` Hz."""

    index: int
    start_sample: int
    end_sample: int
    sample_rate: int = SAMPLE_RATE

    @property
    def start(self) -> float:
        """Début en secondes."""
        return round(self.start_sample / self.sample_rate, 3)

    @property
    def end(self) -> float:
        """Fin en secondes."""
        return round(self.end_sample / self.sample_rate, 3)

    @property
    def duration(self) -> float:
        """Durée en secondes."""
        return round((self.end_sample - self.start_sample) / self.sample_rate, 3)

    @property
    def filename(self) -> str:
        """Nom du fichier WAV du segment (ex. ``segment_0001.wav``)."""
        return SEGMENT_PATTERN.format(index=self.index)

    def to_dict(self) -> dict[str, Any]:
        """Représentation utilisée dans ``metadata.json``."""
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "file": self.filename,
        }


# ------------------------------------------------------------------ #
#  Étapes du pipeline
# ------------------------------------------------------------------ #

def validate_audio_file(audio_path: PathLike) -> Path:
    """Vérifie que le fichier existe, est un fichier et n'est pas vide ; retourne son ``Path``."""
    path = Path(audio_path)
    if not path.exists():
        raise AudioFileNotFoundError(f"Fichier audio introuvable : '{path}'.")
    if not path.is_file():
        raise AudioFileNotFoundError(f"Le chemin ne désigne pas un fichier : '{path}'.")
    if path.stat().st_size == 0:
        raise EmptyAudioFileError(f"Le fichier audio est vide (0 octet) : '{path}'.")
    return path


def build_segments(timestamps: Sequence[dict[str, int]], total_samples: int) -> list[SpeechSegment]:
    """
    Transforme les horodatages Silero VAD (en échantillons) en ``SpeechSegment`` numérotés.

    Les limites sont bornées à la durée de l'audio et les segments vides sont ignorés.
    """
    segments: list[SpeechSegment] = []
    for ts in timestamps:
        start = max(0, int(ts["start"]))
        end = min(total_samples, int(ts["end"]))
        if end > start:
            segments.append(SpeechSegment(index=len(segments) + 1, start_sample=start, end_sample=end))
    return segments


def prepare_output_dir(output_root: PathLike, source: Path) -> Path:
    """
    Crée ``<output_root>/<nom_du_fichier>/`` et supprime les résultats d'un traitement précédent
    (``segment_*.wav`` et ``metadata.json``) pour ne pas mélanger deux exécutions.
    """
    output_dir = Path(output_root) / source.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in [*output_dir.glob("segment_*.wav"), output_dir / METADATA_FILENAME]:
        old_file.unlink(missing_ok=True)
    return output_dir


def build_metadata(source: Path, total_samples: int, segments: Sequence[SpeechSegment]) -> dict[str, Any]:
    """Construit le contenu de ``metadata.json``."""
    return {
        "source": source.name,
        "duration": round(total_samples / SAMPLE_RATE, 3),
        "segments_count": len(segments),
        "sample_rate": SAMPLE_RATE,
        "segments": [segment.to_dict() for segment in segments],
    }


def write_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    """Écrit ``metadata.json`` (UTF-8, indenté) et retourne son chemin."""
    path = output_dir / METADATA_FILENAME
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ #
#  Fonctions publiques
# ------------------------------------------------------------------ #

def detect_speech_segments(audio_path: PathLike, config: Optional[VADConfig] = None) -> list[dict[str, float]]:
    """
    Détecte les portions parlées d'un fichier audio, sans rien écrire sur le disque.

    Retourne ``[{"start": s, "end": s}, ...]`` en secondes (liste vide si aucune parole).
    """
    source = validate_audio_file(audio_path)
    config = config or VADConfig.from_env()
    find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="vad_") as tmp:
        normalized = Path(tmp) / "normalized.wav"
        convert_to_pcm_wav(source, normalized)
        audio = load_pcm_wav(normalized)
        segments = build_segments(detect_speech_timestamps(audio, config), audio.size)
    return [{"start": s.start, "end": s.end} for s in segments]


def identify_silence_intervals(
    speech_segments: Sequence[SpeechSegment | dict[str, Any]],
    total_duration: float,
) -> list[dict[str, float]]:
    """
    Identifie les intervalles de silence situés avant, entre et après les portions de parole.

    Retourne une liste de dictionnaires :
    ``[{"start": float, "end": float, "duration": float}, ...]`` en secondes.
    """
    silences: list[dict[str, float]] = []
    current_time = 0.0

    for seg in speech_segments:
        start = seg.start if isinstance(seg, SpeechSegment) else float(seg["start"])
        end = seg.end if isinstance(seg, SpeechSegment) else float(seg["end"])

        if start > current_time:
            duration = round(start - current_time, 3)
            if duration > 0:
                silences.append({
                    "start": round(current_time, 3),
                    "end": round(start, 3),
                    "duration": duration,
                })
        current_time = max(current_time, end)

    if round(total_duration - current_time, 3) > 0:
        silences.append({
            "start": round(current_time, 3),
            "end": round(total_duration, 3),
            "duration": round(total_duration - current_time, 3),
        })

    return silences


def detect_silence_intervals(
    audio_path: PathLike,
    config: Optional[VADConfig] = None,
) -> list[dict[str, float]]:
    """
    Détecte les intervalles de silence d'un fichier audio, sans rien écrire sur le disque.

    Retourne ``[{"start": s, "end": s, "duration": s}, ...]`` en secondes.
    """
    source = validate_audio_file(audio_path)
    config = config or VADConfig.from_env()
    find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="vad_") as tmp:
        normalized = Path(tmp) / "normalized.wav"
        convert_to_pcm_wav(source, normalized)
        audio = load_pcm_wav(normalized)
        speech_segments = build_segments(detect_speech_timestamps(audio, config), audio.size)
        total_duration = round(audio.size / SAMPLE_RATE, 3)
        return identify_silence_intervals(speech_segments, total_duration)


def split_audio_on_silences(
    audio_path: PathLike,
    output_root: PathLike = DEFAULT_OUTPUT_ROOT,
    config: Optional[VADConfig] = None,
) -> dict[str, Any]:
    """
    Découpe un fichier audio en segments WAV de parole et écrit ``metadata.json``.

    Paramètres
    ----------
    audio_path : chemin du fichier audio (tout format lisible par FFmpeg).
    output_root : dossier racine des résultats (``outputs/audio`` par défaut).
    config : paramètres VAD ; par défaut ``VADConfig.from_env()``.

    Retourne le dictionnaire écrit dans ``metadata.json``, complété par
    ``output_dir`` (chemin du dossier des résultats).

    Lève une sous-classe de ``VoiceProcessingError`` en cas d'échec,
    notamment ``NoSpeechDetectedError`` si aucune parole n'est trouvée
    (aucun dossier de résultats n'est alors créé).
    """
    source = validate_audio_file(audio_path)
    config = config or VADConfig.from_env()
    find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="vad_") as tmp:
        normalized = Path(tmp) / "normalized.wav"
        convert_to_pcm_wav(source, normalized)
        audio = load_pcm_wav(normalized)
        segments = build_segments(detect_speech_timestamps(audio, config), audio.size)
        if not segments:
            raise NoSpeechDetectedError(f"Aucune parole détectée dans '{source.name}'.")
        output_dir = prepare_output_dir(output_root, source)
        for segment in segments:
            extract_segment(normalized, output_dir / segment.filename, segment.start_sample, segment.end_sample)
    metadata = build_metadata(source, audio.size, segments)
    write_metadata(output_dir, metadata)
    return {**metadata, "output_dir": str(output_dir)}


# ------------------------------------------------------------------ #
#  Ligne de commande
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Découpe un fichier audio sur les silences (FFmpeg + Silero VAD).")
    parser.add_argument("audio", help="Fichier audio à traiter.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Dossier racine des résultats.")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-speech-duration-ms", type=int)
    parser.add_argument("--min-silence-duration-ms", type=int)
    parser.add_argument("--speech-pad-ms", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée CLI ; retourne 0 en cas de succès, 1 en cas d'erreur."""
    args = _parse_args(argv)
    try:
        config = VADConfig.from_env().with_overrides(
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_duration_ms,
            min_silence_duration_ms=args.min_silence_duration_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
        result = split_audio_on_silences(args.audio, args.output_dir, config)
    except VoiceProcessingError as exc:
        print(f"Erreur ({type(exc).__name__}) : {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
