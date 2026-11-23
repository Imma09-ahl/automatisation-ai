"""
Configuration de la détection de parole (Silero VAD).

Les valeurs peuvent être fournies :
  - directement : ``VADConfig(threshold=0.6)`` ;
  - par variables d'environnement : ``VAD_THRESHOLD``, ``VAD_MIN_SPEECH_DURATION_MS``,
    ``VAD_MIN_SILENCE_DURATION_MS``, ``VAD_SPEECH_PAD_MS`` (voir ``VADConfig.from_env``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping, Optional

from app.audio_errors import InvalidVADConfigError

SAMPLE_RATE = 16000
"""Fréquence d'échantillonnage attendue par Silero VAD et utilisée pour les segments."""

ENV_PREFIX = "VAD_"


@dataclass(frozen=True)
class VADConfig:
    """
    Paramètres de détection transmis à ``silero_vad.get_speech_timestamps``.

    Attributs
    ---------
    threshold : float
        Probabilité (0 à 1, bornes exclues) au-dessus de laquelle une fenêtre est
        considérée comme de la parole.
    min_speech_duration_ms : int
        Les portions parlées plus courtes sont ignorées (bruits brefs).
    min_silence_duration_ms : int
        Durée minimale d'un silence pour séparer deux portions parlées.
    speech_pad_ms : int
        Marge ajoutée de chaque côté d'une portion parlée.
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 100

    def __post_init__(self) -> None:
        _validate_threshold(self.threshold)
        for name in ("min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms"):
            _validate_non_negative_int(name, getattr(self, name))

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "VADConfig":
        """
        Construit une configuration à partir des variables d'environnement ``VAD_*``.

        Les variables absentes conservent leur valeur par défaut.
        """
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field in fields(cls):
            raw = env.get(ENV_PREFIX + field.name.upper())
            if raw is not None and raw.strip() != "":
                values[field.name] = _parse_value(field.name, raw, float if field.type in (float, "float") else int)
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "VADConfig":
        """Retourne une copie où seuls les paramètres non ``None`` sont remplacés."""
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise InvalidVADConfigError(f"Paramètre(s) VAD inconnu(s) : {', '.join(sorted(unknown))}")
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        """Retourne les paramètres sous forme de dictionnaire (pour le JSON)."""
        return asdict(self)


def _parse_value(name: str, raw: str, kind: type) -> Any:
    try:
        return kind(raw.strip())
    except ValueError as exc:
        raise InvalidVADConfigError(
            f"Valeur invalide pour {ENV_PREFIX}{name.upper()} : '{raw}' ({kind.__name__} attendu)."
        ) from exc


def _validate_threshold(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise InvalidVADConfigError(f"threshold doit être un nombre strictement entre 0 et 1 (reçu : {value!r}).")


def _validate_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidVADConfigError(f"{name} doit être un entier positif ou nul (reçu : {value!r}).")
