"""
Accès au modèle Silero VAD.

Le modèle est chargé une seule fois, au premier besoin (chargement différé),
puis réutilisé. Les poids sont fournis par le paquet ``silero-vad`` lui-même :
aucun téléchargement n'est effectué.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from app.audio_errors import VADDetectionError
from app.vad_config import SAMPLE_RATE, VADConfig

_model: Any = None
_load_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_vad_model() -> Any:
    """Retourne le modèle Silero VAD, en le chargeant au premier appel uniquement."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                try:
                    _model = load_silero_vad()
                except Exception as exc:
                    raise VADDetectionError(f"Impossible de charger le modèle Silero VAD : {exc}") from exc
    return _model


def reset_vad_model() -> None:
    """Oublie le modèle chargé (utile pour les tests)."""
    global _model
    with _load_lock:
        _model = None


def detect_speech_timestamps(
    audio: np.ndarray,
    config: VADConfig,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict[str, int]]:
    """
    Détecte les portions parlées d'un signal mono float32 normalisé dans [-1, 1].

    Retourne une liste ``[{"start": n, "end": m}, ...]`` exprimée en échantillons,
    marge ``speech_pad_ms`` comprise, telle que calculée par Silero VAD.
    """
    model = get_vad_model()
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    try:
        # Le modèle garde un état interne : une seule inférence à la fois.
        with _inference_lock:
            timestamps = get_speech_timestamps(
                tensor,
                model,
                threshold=config.threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=config.min_speech_duration_ms,
                min_silence_duration_ms=config.min_silence_duration_ms,
                speech_pad_ms=config.speech_pad_ms,
                return_seconds=False,
            )
    except Exception as exc:
        raise VADDetectionError(f"Échec de la détection de parole par Silero VAD : {exc}") from exc
    return [{"start": int(ts["start"]), "end": int(ts["end"])} for ts in timestamps]
