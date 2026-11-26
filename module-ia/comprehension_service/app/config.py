"""
Configuration du Comprehension Service.

Gère les variables d'environnement pour :
  - Les seuils de rapprochement flou RapidFuzz (score global et score partiel)
  - Le répertoire des dictionnaires par langue
  - Les paramètres du LLM (Instructor + OpenAI / API compatible)
  - L'activation ou désactivation explicite du LLM
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ENV_PREFIX = "COMPREHENSION_"


@dataclass(frozen=True)
class ComprehensionConfig:
    """Paramètres configurables du service compréhension."""

    # Seuils RapidFuzz (0 à 100) - recommandation 80.0 par défaut
    fuzzy_match_threshold: float = 80.0
    fuzzy_partial_threshold: float = 85.0

    # Dictionnaires
    dictionaries_dir: Path = Path(__file__).resolve().parent.parent / "data" / "dictionaries"

    # LLM & Instructor
    llm_enabled: bool = True
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "ComprehensionConfig":
        """Charge la configuration depuis les variables d'environnement."""
        api_key = os.getenv("LLM_API_KEY")
        # Le LLM est activé seulement si une clé est explicitement renseignée ou forcée par LLM_ENABLED
        llm_enabled_env = os.getenv("LLM_ENABLED", "true" if api_key else "false").lower()
        llm_enabled = llm_enabled_env in ("1", "true", "yes", "on")

        dict_dir_raw = os.getenv("DICTIONARIES_DIR")
        dict_dir = Path(dict_dir_raw) if dict_dir_raw else (Path(__file__).resolve().parent.parent / "data" / "dictionaries")

        return cls(
            fuzzy_match_threshold=float(os.getenv("FUZZY_MATCH_THRESHOLD", os.getenv("COMPREHENSION_FUZZY_MATCH_THRESHOLD", "80.0"))),
            fuzzy_partial_threshold=float(os.getenv("FUZZY_PARTIAL_THRESHOLD", os.getenv("COMPREHENSION_FUZZY_PARTIAL_THRESHOLD", "85.0"))),
            dictionaries_dir=dict_dir,
            llm_enabled=llm_enabled and bool(api_key),
            llm_api_key=api_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0")),
        )
