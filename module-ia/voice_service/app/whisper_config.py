"""
Configuration de la transcription française (faster-whisper).

Les valeurs peuvent être fournies :
  - directement : ``WhisperConfig(model_size="base")`` ;
  - par variables d'environnement ``WHISPER_*`` (voir ``WhisperConfig.from_env``) :
    ``WHISPER_MODEL_SIZE``, ``WHISPER_DEVICE``, ``WHISPER_COMPUTE_TYPE``, ``WHISPER_BEAM_SIZE``,
    ``WHISPER_CPU_THREADS``, ``WHISPER_DOWNLOAD_ROOT``, ``WHISPER_LOCAL_FILES_ONLY``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from app.audio_errors import InvalidWhisperConfigError

ENV_PREFIX = "WHISPER_"
LANGUAGE = "fr"
"""Langue imposée à Whisper : pas de détection automatique."""

SUPPORTED_DEVICES = ("cpu", "cuda", "auto")

_TRUE_VALUES = ("1", "true", "yes", "oui", "on")
_FALSE_VALUES = ("0", "false", "no", "non", "off")


@dataclass(frozen=True)
class WhisperConfig:
    """
    Paramètres de chargement du modèle faster-whisper et de transcription.

    Attributs
    ---------
    model_size : str
        Taille du modèle (``tiny``, ``base``, ``small``, ``medium``, ``large-v3``...)
        ou chemin d'un dossier contenant un modèle CTranslate2 déjà converti.
    device : str
        ``cpu``, ``cuda`` (carte NVIDIA) ou ``auto``.
    compute_type : str
        Précision des calculs ; ``int8`` est le plus rapide et le plus léger sur processeur.
    beam_size : int
        Largeur du faisceau de recherche : plus grand = un peu plus précis, mais plus lent.
    cpu_threads : int
        Nombre de threads processeur (``0`` = choix automatique).
    download_root : str | None
        Dossier de stockage des poids (``None`` = cache Hugging Face de l'utilisateur).
    local_files_only : bool
        ``True`` interdit tout téléchargement : le modèle doit déjà être présent.
    """

    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    cpu_threads: int = 0
    download_root: Optional[str] = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        _validate_model_size(self.model_size)
        _validate_choice("device", self.device, SUPPORTED_DEVICES)
        _validate_non_empty_str("compute_type", self.compute_type)
        _validate_int("beam_size", self.beam_size, minimum=1)
        _validate_int("cpu_threads", self.cpu_threads, minimum=0)
        if self.download_root is not None:
            _validate_non_empty_str("download_root", self.download_root)
        if not isinstance(self.local_files_only, bool):
            raise InvalidWhisperConfigError(f"local_files_only doit être un booléen (reçu : {self.local_files_only!r}).")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "WhisperConfig":
        """Construit une configuration à partir des variables ``WHISPER_*`` ; les absentes gardent leur défaut."""
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field in fields(cls):
            variable = ENV_PREFIX + field.name.upper()
            raw = env.get(variable)
            if raw is not None and raw.strip() != "":
                values[field.name] = _parse_value(variable, raw.strip(), _PARSERS.get(field.name, str))
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "WhisperConfig":
        """Retourne une copie où seuls les paramètres non ``None`` sont remplacés."""
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise InvalidWhisperConfigError(f"Paramètre(s) Whisper inconnu(s) : {', '.join(sorted(unknown))}")
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def model_key(self) -> tuple:
        """Paramètres qui imposent de recharger le modèle lorsqu'ils changent."""
        return (self.model_size, self.device, self.compute_type, self.cpu_threads, self.download_root, self.local_files_only)

    def to_dict(self) -> dict[str, Any]:
        """Retourne les paramètres sous forme de dictionnaire."""
        return asdict(self)


def _parse_bool(raw: str) -> bool:
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(raw)


_PARSERS: dict[str, Callable[[str], Any]] = {
    "beam_size": int,
    "cpu_threads": int,
    "local_files_only": _parse_bool,
}


def _parse_value(variable: str, raw: str, parser: Callable[[str], Any]) -> Any:
    try:
        return parser(raw)
    except ValueError as exc:
        expected = "booléen (true/false)" if parser is _parse_bool else getattr(parser, "__name__", "valeur")
        raise InvalidWhisperConfigError(f"Valeur invalide pour {variable} : '{raw}' ({expected} attendu).") from exc


def _validate_model_size(value: Any) -> None:
    from faster_whisper import available_models

    _validate_non_empty_str("model_size", value)
    if value not in available_models() and not Path(value).is_dir():
        raise InvalidWhisperConfigError(
            f"Modèle Whisper inconnu : '{value}'. Valeurs possibles : {', '.join(available_models())}, "
            "ou le chemin d'un dossier contenant un modèle CTranslate2."
        )


def _validate_choice(name: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise InvalidWhisperConfigError(f"{name} doit valoir {', '.join(choices)} (reçu : {value!r}).")


def _validate_non_empty_str(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidWhisperConfigError(f"{name} doit être une chaîne non vide (reçu : {value!r}).")


def _validate_int(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidWhisperConfigError(f"{name} doit être un entier supérieur ou égal à {minimum} (reçu : {value!r}).")
