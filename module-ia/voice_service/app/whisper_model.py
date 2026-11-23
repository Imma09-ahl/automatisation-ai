"""
Accès au modèle faster-whisper.

Le modèle est chargé au premier besoin (chargement différé), puis réutilisé tant que
les paramètres de chargement ne changent pas. Au premier chargement, faster-whisper
télécharge les poids depuis Hugging Face (sauf si ``local_files_only`` est activé).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from faster_whisper import WhisperModel

from app.audio_errors import TranscriptionError, WhisperModelLoadError
from app.whisper_config import LANGUAGE, WhisperConfig

_model: Any = None
_model_key: Optional[tuple] = None
_load_lock = threading.Lock()


def get_whisper_model(config: WhisperConfig) -> Any:
    """Retourne le modèle correspondant à ``config``, en le chargeant seulement si nécessaire."""
    global _model, _model_key
    key = config.model_key()
    with _load_lock:
        if _model is None or _model_key != key:
            _model, _model_key = None, None
            _model = _load(config)
            _model_key = key
        return _model


def reset_whisper_model() -> None:
    """Oublie le modèle chargé (utile pour les tests)."""
    global _model, _model_key
    with _load_lock:
        _model, _model_key = None, None


def transcribe_file(audio_path: Path, config: WhisperConfig) -> list[dict[str, Any]]:
    """
    Transcrit un fichier audio en français.

    Retourne ``[{"start": s, "end": s, "text": "..."}, ...]`` ; les temps sont relatifs
    au début du fichier. La détection de parole interne de faster-whisper est désactivée :
    le fichier est déjà un segment de parole découpé par Silero VAD.
    """
    model = get_whisper_model(config)
    try:
        pieces, _info = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            beam_size=config.beam_size,
            vad_filter=False,
        )
        # Les résultats sont produits à la demande : la transcription a lieu pendant ce parcours.
        return [{"start": float(p.start), "end": float(p.end), "text": p.text.strip()} for p in pieces]
    except Exception as exc:
        raise TranscriptionError(f"Échec de la transcription de '{Path(audio_path).name}' : {exc}") from exc


def _load(config: WhisperConfig) -> Any:
    try:
        return WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
            download_root=config.download_root,
            local_files_only=config.local_files_only,
        )
    except Exception as exc:
        hint = (
            " Le modèle n'est pas présent localement : désactivez local_files_only pour le télécharger."
            if config.local_files_only
            else " Vérifiez la connexion internet pour le premier téléchargement des poids."
        )
        raise WhisperModelLoadError(f"Impossible de charger le modèle faster-whisper '{config.model_size}' : {exc}.{hint}") from exc
