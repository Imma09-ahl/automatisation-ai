"""
Moteur de transcription française basé sur faster-whisper.

Configuration requise :
  - WhisperModel
  - language="fr"
  - task="transcribe"
  - device="cpu"
  - compute_type="int8"
  - modèle configurable (défaut : "base")
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from faster_whisper import WhisperModel

from app.audio_errors import TranscriptionError, WhisperModelLoadError

_whisper_instance: Any = None
_loaded_model_size: Optional[str] = None
_load_lock = threading.Lock()


def get_whisper_engine(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
) -> Any:
    """Charge ou retourne le modèle faster-whisper (singleton par configuration)."""
    global _whisper_instance, _loaded_model_size
    with _load_lock:
        if _whisper_instance is None or _loaded_model_size != model_size:
            try:
                _whisper_instance = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=download_root,
                    local_files_only=local_files_only,
                )
                _loaded_model_size = model_size
            except Exception as exc:
                raise WhisperModelLoadError(
                    f"Impossible de charger le modèle faster-whisper '{model_size}' : {exc}"
                ) from exc
    return _whisper_instance


def reset_whisper_engine() -> None:
    """Réinitialise l'instance du modèle (utile pour les tests)."""
    global _whisper_instance, _loaded_model_size
    with _load_lock:
        _whisper_instance = None
        _loaded_model_size = None


def transcribe_french_segment(
    audio_path: Path | str,
    model_size: str = "base",
    beam_size: int = 5,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Transcrit un segment audio WAV mono 16 kHz en français.

    Retourne une liste de morceaux horodatés (relatifs au segment) :
    [{"start": float, "end": float, "text": str}]
    """
    path = Path(audio_path)
    model = get_whisper_engine(model_size=model_size, local_files_only=local_files_only)
    try:
        segments_gen, _info = model.transcribe(
            str(path),
            language="fr",
            task="transcribe",
            beam_size=beam_size,
            vad_filter=False,  # Silero VAD a déjà découpé la parole
        )
        results: list[dict[str, Any]] = []
        for s in segments_gen:
            text = s.text.strip()
            if text:
                results.append({
                    "start": round(float(s.start), 3),
                    "end": round(float(s.end), 3),
                    "text": text,
                })
        return results
    except Exception as exc:
        raise TranscriptionError(f"Échec de transcription faster-whisper pour '{path.name}' : {exc}") from exc
