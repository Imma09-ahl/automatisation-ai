"""
Transcription française des segments de parole avec faster-whisper.

Pipeline :
  1. Lecture du ``metadata.json`` produit par la détection des silences.
  2. Transcription en français de chaque ``segment_XXXX.wav``.
  3. Ajout du texte de chaque segment et du texte complet dans ``metadata.json``.
  4. Écriture du texte complet dans ``transcription.txt``.

``metadata.json`` n'est réécrit qu'une fois tous les segments transcrits : en cas
d'échec, les résultats de la détection des silences restent intacts.

Utilisation en ligne de commande (depuis ``voice_service/``) :
    python -m app.transcription_fr ../examples/audio/demo_parole.wav --output-dir ../outputs/audio
    python -m app.transcription_fr --segments-dir ../outputs/audio/demo_parole
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from app.audio_errors import EmptyTranscriptionError, MetadataError, VoiceProcessingError
from app.silence_detection import DEFAULT_OUTPUT_ROOT, METADATA_FILENAME, split_audio_on_silences, write_metadata
from app.vad_config import VADConfig
from app.whisper_config import LANGUAGE, WhisperConfig
from app.whisper_model import transcribe_file

PathLike = Union[str, Path]

TRANSCRIPT_FILENAME = "transcription.txt"
ENGINE = "faster-whisper"


# ------------------------------------------------------------------ #
#  Étapes
# ------------------------------------------------------------------ #

def load_metadata(output_dir: Path) -> dict[str, Any]:
    """Lit et vérifie le ``metadata.json`` d'un dossier de segments."""
    path = output_dir / METADATA_FILENAME
    if not path.is_file():
        raise MetadataError(
            f"'{METADATA_FILENAME}' introuvable dans '{output_dir}'. Lancez d'abord la détection des silences."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MetadataError(f"'{METADATA_FILENAME}' illisible dans '{output_dir}' : {exc}") from exc
    segments = metadata.get("segments") if isinstance(metadata, dict) else None
    if not isinstance(segments, list) or not segments:
        raise MetadataError(f"'{METADATA_FILENAME}' ne contient aucun segment dans '{output_dir}'.")
    for segment in segments:
        if not isinstance(segment, dict) or not {"index", "start", "end", "file"} <= segment.keys():
            raise MetadataError(f"Segment mal formé dans '{METADATA_FILENAME}' : {segment!r}")
    return metadata


def resolve_segment_files(output_dir: Path, segments: Sequence[dict[str, Any]]) -> list[Path]:
    """Retourne le chemin de chaque fichier de segment, en vérifiant qu'il existe."""
    paths = [output_dir / str(segment["file"]) for segment in segments]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise MetadataError(f"Fichier(s) de segment manquant(s) dans '{output_dir}' : {', '.join(missing)}")
    return paths


def join_texts(texts: Iterable[str]) -> str:
    """Assemble des morceaux de texte en ignorant les morceaux vides."""
    return " ".join(text.strip() for text in texts if text and text.strip())


def transcribe_segment(path: Path, config: WhisperConfig) -> str:
    """Transcrit un fichier de segment et retourne son texte (chaîne vide si rien n'est reconnu)."""
    return join_texts(piece["text"] for piece in transcribe_file(path, config))


def build_transcription(metadata: dict[str, Any], texts: Sequence[str], config: WhisperConfig) -> dict[str, Any]:
    """Ajoute le texte de chaque segment et le texte complet aux métadonnées existantes."""
    segments = [{**segment, "text": text} for segment, text in zip(metadata["segments"], texts)]
    return {
        **metadata,
        "segments": segments,
        "transcription": {
            "language": LANGUAGE,
            "engine": ENGINE,
            "model": config.model_size,
            "text": join_texts(texts),
        },
    }


def write_transcript(output_dir: Path, text: str) -> Path:
    """Écrit le texte complet dans ``transcription.txt`` (UTF-8) et retourne son chemin."""
    path = output_dir / TRANSCRIPT_FILENAME
    path.write_text(text + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ #
#  API publique
# ------------------------------------------------------------------ #

def transcribe_segments_dir(output_dir: PathLike, config: Optional[WhisperConfig] = None) -> dict[str, Any]:
    """
    Transcrit en français les segments d'un dossier produit par ``split_audio_on_silences``.

    Retourne le contenu du nouveau ``metadata.json``, complété par ``output_dir``.
    Lève ``EmptyTranscriptionError`` si aucun texte n'est reconnu dans aucun segment,
    sans modifier ``metadata.json``.
    """
    output_dir = Path(output_dir)
    config = config or WhisperConfig.from_env()
    metadata = load_metadata(output_dir)
    paths = resolve_segment_files(output_dir, metadata["segments"])
    texts = [transcribe_segment(path, config) for path in paths]
    if not join_texts(texts):
        raise EmptyTranscriptionError(f"Aucun texte reconnu dans les {len(paths)} segment(s) de '{output_dir.name}'.")
    result = build_transcription(metadata, texts, config)
    write_metadata(output_dir, result)
    write_transcript(output_dir, result["transcription"]["text"])
    return {**result, "output_dir": str(output_dir)}


def split_and_transcribe_fr(
    audio_path: PathLike,
    output_root: PathLike = DEFAULT_OUTPUT_ROOT,
    vad_config: Optional[VADConfig] = None,
    whisper_config: Optional[WhisperConfig] = None,
) -> dict[str, Any]:
    """Découpe un fichier audio sur les silences, puis transcrit chaque segment en français."""
    stale_transcript = Path(output_root) / Path(audio_path).stem / TRANSCRIPT_FILENAME
    stale_transcript.unlink(missing_ok=True)
    split = split_audio_on_silences(audio_path, output_root, vad_config)
    return transcribe_segments_dir(split["output_dir"], whisper_config)


# ------------------------------------------------------------------ #
#  Ligne de commande
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcription française (faster-whisper) des segments de parole.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("audio", nargs="?", help="Fichier audio à découper puis transcrire.")
    source.add_argument("--segments-dir", help="Dossier de segments déjà découpés (contenant metadata.json).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Dossier racine des résultats.")
    parser.add_argument("--model-size")
    parser.add_argument("--device")
    parser.add_argument("--compute-type")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--local-files-only", action="store_const", const=True, help="Interdit le téléchargement du modèle.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée CLI ; retourne 0 en cas de succès, 1 en cas d'erreur."""
    args = _parse_args(argv)
    try:
        config = WhisperConfig.from_env().with_overrides(
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            local_files_only=args.local_files_only,
        )
        if args.segments_dir:
            result = transcribe_segments_dir(args.segments_dir, config)
        else:
            result = split_and_transcribe_fr(args.audio, args.output_dir, VADConfig.from_env(), config)
    except VoiceProcessingError as exc:
        print(f"Erreur ({type(exc).__name__}) : {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
