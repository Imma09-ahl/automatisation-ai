"""
Transcription audio multilingue (Français, Fon, Yoruba) pour le service voix.

Ce module orchestre :
  1. La validation et normalisation de la langue demandée ('fr', 'fon', 'yo'/'yor').
  2. L'orientation vers le moteur adéquat :
     - Français : faster-whisper (modèle 'base' par défaut)
     - Fon : Meta MMS (facebook/mms-1b-all, adaptateur 'fon')
     - Yoruba : Meta MMS (facebook/mms-1b-all, adaptateur 'yor')
  3. La lecture des segments audio issus de Silero VAD (metadata.json).
  4. La transcription de chaque segment et le calcul des horodatages globaux.
  5. La génération et la sauvegarde du fichier transcription.json unifié.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union

_voice_service_dir = Path(__file__).resolve().parent.parent
if str(_voice_service_dir) not in sys.path:
    sys.path.insert(0, str(_voice_service_dir))

from app.audio_errors import (
    AudioFileNotFoundError,
    EmptyTranscriptionError,
    MetadataError,
    UnsupportedLanguageError,
    VoiceProcessingError,
)
from app.faster_whisper_engine import transcribe_french_segment
from app.mms_engine import MMS_MODEL_ID, transcribe_mms_segment
from app.silence_detection import (
    DEFAULT_OUTPUT_ROOT,
    METADATA_FILENAME,
    split_audio_on_silences,
    validate_audio_file,
)

PathLike = Union[str, Path]
TRANSCRIPTION_FILENAME = "transcription.json"

LANGUAGE_MAP = {
    "fr": "fr",
    "fra": "fr",
    "fon": "fon",
    "yo": "yor",
    "yor": "yor",
}

ENGINE_MAP = {
    "fr": ("faster-whisper", "base"),
    "fon": ("mms-1b-all", MMS_MODEL_ID),
    "yor": ("mms-1b-all", MMS_MODEL_ID),
}


def normalize_language_code(language: str) -> str:
    """
    Normalise le code langue d'entrée vers son identifiant canonique :
      - 'fr' -> 'fr'
      - 'fon' -> 'fon'
      - 'yo' ou 'yor' -> 'yor'

    Lève ``UnsupportedLanguageError`` si la langue n'est pas supportée.
    """
    if not isinstance(language, str):
        raise UnsupportedLanguageError(f"Le code langue doit être une chaîne (reçu: {language!r}).")
    cleaned = language.strip().lower()
    if cleaned not in LANGUAGE_MAP:
        supported = sorted(set(LANGUAGE_MAP.keys()))
        raise UnsupportedLanguageError(
            f"Langue non prise en charge : '{language}'. Langues supportées : {supported}"
        )
    return LANGUAGE_MAP[cleaned]


def load_vad_metadata(output_dir: Path) -> dict[str, Any]:
    """Lit et valide le fichier metadata.json produit par Silero VAD."""
    meta_path = output_dir / METADATA_FILENAME
    if not meta_path.is_file():
        raise MetadataError(
            f"'{METADATA_FILENAME}' introuvable dans '{output_dir}'. Découpez d'abord l'audio avec Silero VAD."
        )
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MetadataError(f"'{METADATA_FILENAME}' illisible dans '{output_dir}' : {exc}") from exc

    segments = metadata.get("segments")
    if not isinstance(segments, list) or len(segments) == 0:
        raise MetadataError(f"'{METADATA_FILENAME}' ne contient aucun segment de parole dans '{output_dir}'.")

    for seg in segments:
        if not isinstance(seg, dict) or not {"index", "start", "end", "file"} <= seg.keys():
            raise MetadataError(f"Structure de segment invalide dans '{METADATA_FILENAME}' : {seg!r}")

    return metadata


def transcribe_single_segment(
    segment_file: Path,
    lang_code: str,
    whisper_model: str = "base",
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """Transcrit un segment individuel selon la langue spécifiée."""
    if not segment_file.is_file():
        raise AudioFileNotFoundError(f"Fichier de segment audio introuvable : '{segment_file}'.")

    if lang_code == "fr":
        return transcribe_french_segment(
            segment_file,
            model_size=whisper_model,
            local_files_only=local_files_only,
        )
    elif lang_code in ("fon", "yor"):
        return transcribe_mms_segment(
            segment_file,
            adapter_lang=lang_code,
            local_files_only=local_files_only,
        )
    else:
        raise UnsupportedLanguageError(f"Langue non gérée : '{lang_code}'.")


def compute_global_timestamps(
    segment_offset: float,
    pieces: Sequence[dict[str, Any]],
    segment_end: float,
) -> list[dict[str, Any]]:
    """
    Ajuste les horodatages relatifs d'un segment en horodatages globaux par rapport
    au fichier audio source original.
    """
    if not pieces:
        return []
    adjusted: list[dict[str, Any]] = []
    for p in pieces:
        p_start = round(segment_offset + float(p.get("start", 0.0)), 3)
        p_end = round(segment_offset + float(p.get("end", segment_end - segment_offset)), 3)
        # S'assurer de ne pas dépasser la borne du segment
        p_end = min(p_end, segment_end)
        adjusted.append({
            "start": p_start,
            "end": p_end,
            "text": str(p.get("text", "")).strip(),
        })
    return adjusted


def transcribe_segments_dir(
    segments_dir: PathLike,
    language: str,
    whisper_model: str = "base",
    local_files_only: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """
    Transcrit l'ensemble des segments présents dans un dossier généré par Silero VAD.

    Paramètres
    ----------
    segments_dir : dossier contenant ``metadata.json`` et les fichiers ``segment_XXXX.wav``.
    language : code langue ('fr', 'fon', 'yo', 'yor').
    whisper_model : nom du modèle pour le français (défaut 'base').
    local_files_only : n'autorise que les modèles déjà en cache local.
    allow_empty : si False, lève ``EmptyTranscriptionError`` si aucun texte n'est extrait.

    Retourne le dictionnaire correspondant au schéma unifié ``transcription.json``.
    """
    dir_path = Path(segments_dir)
    if not dir_path.is_dir():
        raise AudioFileNotFoundError(f"Le dossier de segments spécifié est introuvable : '{dir_path}'.")

    canonical_lang = normalize_language_code(language)
    engine_name, default_model_name = ENGINE_MAP[canonical_lang]
    model_name = whisper_model if canonical_lang == "fr" else default_model_name

    metadata = load_vad_metadata(dir_path)
    segments = metadata["segments"]

    processed_segments: list[dict[str, Any]] = []
    all_texts: list[str] = []

    for seg in segments:
        seg_file = dir_path / seg["file"]
        if not seg_file.is_file():
            raise MetadataError(f"Fichier de segment manquant : '{seg_file.name}' dans '{dir_path}'.")

        pieces = transcribe_single_segment(
            seg_file,
            canonical_lang,
            whisper_model=model_name,
            local_files_only=local_files_only,
        )

        global_pieces = compute_global_timestamps(
            segment_offset=float(seg["start"]),
            pieces=pieces,
            segment_end=float(seg["end"]),
        )

        seg_text = " ".join(p["text"] for p in global_pieces if p["text"]).strip()
        if not seg_text and pieces:
            seg_text = " ".join(str(p.get("text", "")).strip() for p in pieces).strip()

        if seg_text:
            all_texts.append(seg_text)

        processed_segments.append({
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
            "file": seg["file"],
            "text": seg_text,
        })

    full_text = " ".join(all_texts).strip()
    if not full_text and not allow_empty:
        raise EmptyTranscriptionError(
            f"Aucun texte n'a pu être extrait des {len(segments)} segment(s) dans la langue '{canonical_lang}'."
        )

    result = {
        "source": metadata.get("source", dir_path.name),
        "language": canonical_lang,
        "engine": engine_name,
        "model": model_name,
        "text": full_text,
        "segments_count": len(processed_segments),
        "segments": processed_segments,
    }

    # Écriture du fichier transcription.json
    out_json = dir_path / TRANSCRIPTION_FILENAME
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def transcribe(
    audio_path: PathLike,
    language: str = "fr",
    output_root: PathLike = DEFAULT_OUTPUT_ROOT,
    whisper_model: str = "base",
    local_files_only: bool = False,
    allow_empty: bool = False,
    segments: Optional[Sequence[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Point d'entrée complet : découpe automatique des silences puis transcription
    dans la langue demandée ('fr', 'fon', 'yo'/'yor').
    """
    path = validate_audio_file(audio_path)
    canonical_lang = normalize_language_code(language)

    # 1. Découpage sur les silences avec Silero VAD
    vad_res = split_audio_on_silences(path, output_root=output_root)
    output_dir = Path(vad_res["output_dir"])

    # 2. Transcription des segments obtenus
    return transcribe_segments_dir(
        segments_dir=output_dir,
        language=canonical_lang,
        whisper_model=whisper_model,
        local_files_only=local_files_only,
        allow_empty=allow_empty,
    )


# ------------------------------------------------------------------ #
#  Ligne de commande
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcription trilingue (Français, Fon, Yoruba) des segments audio Silero VAD."
    )
    parser.add_argument("target", help="Chemin du fichier audio original OU du dossier contenant metadata.json.")
    parser.add_argument(
        "-l", "--language",
        default="fr",
        help="Langue de transcription : 'fr' (français), 'fon' (fon), 'yo' ou 'yor' (yoruba). Défaut : fr.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Dossier racine des résultats.")
    parser.add_argument("--whisper-model", default="base", help="Modèle faster-whisper pour le français (défaut : base).")
    parser.add_argument("--local-files-only", action="store_true", help="Interdit tout téléchargement de modèle.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    target = Path(args.target)
    try:
        if target.is_dir():
            res = transcribe_segments_dir(
                segments_dir=target,
                language=args.language,
                whisper_model=args.whisper_model,
                local_files_only=args.local_files_only,
            )
        else:
            res = transcribe(
                audio_path=target,
                language=args.language,
                output_root=args.output_dir,
                whisper_model=args.whisper_model,
                local_files_only=args.local_files_only,
            )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    except VoiceProcessingError as exc:
        print(f"Erreur ({type(exc).__name__}) : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
