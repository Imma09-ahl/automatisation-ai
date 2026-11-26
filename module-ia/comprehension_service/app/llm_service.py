"""
Service d'appel conditionnel au LLM via Instructor.

RÈGLE MAÎTRESSE :
  Le LLM n'est JAMAIS appelé systématiquement.
  Il n'est sollicité que si un ou plusieurs champs indispensables
  sont manquants après la phase de rapprochement lexical RapidFuzz.

Fonctionnalités :
  - Utilisation d'Instructor pour contraindre les réponses du LLM au schéma Pydantic.
  - Préservation stricte des informations déjà validées par le dictionnaire.
  - Gestion gracieuse des erreurs : timeout, API indisponible, clé absente, etc.
  - Si le LLM échoue ou est désactivé, les données déjà résolues restent intactes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import instructor
from openai import OpenAI

from app.config import ComprehensionConfig
from app.schemas import FieldMatch, FieldSource, LLMExtractionPayload

logger = logging.getLogger(__name__)

_instructor_client: Optional[instructor.Instructor] = None


def get_instructor_client(config: Optional[ComprehensionConfig] = None) -> Optional[instructor.Instructor]:
    """
    Initialise et retourne le client Instructor, ou None si le LLM est désactivé / clé absente.
    """
    global _instructor_client
    cfg = config or ComprehensionConfig.from_env()

    if not cfg.llm_enabled or not cfg.llm_api_key:
        return None

    if _instructor_client is None:
        try:
            openai_client = OpenAI(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                timeout=cfg.llm_timeout_seconds,
            )
            _instructor_client = instructor.from_openai(openai_client)
        except Exception as exc:
            logger.warning("Échec de l'initialisation du client LLM : %s", exc)
            return None

    return _instructor_client


def reset_instructor_client() -> None:
    """Réinitialise l'instance du client Instructor (utile pour les tests)."""
    global _instructor_client
    _instructor_client = None


def build_targeted_prompt(
    raw_text: str,
    language: str,
    missing_fields: list[str],
    resolved_fields: dict[str, FieldMatch],
    missing_intent: bool = False,
) -> str:
    """
    Construit un prompt ultra-ciblé informant le LLM des champs manquants à extraire.
    """
    resolved_info = []
    for k, v in resolved_fields.items():
        if v.is_resolved:
            resolved_info.append(f"  - {k}: {v.value}")

    resolved_str = "\n".join(resolved_info) if resolved_info else "  (aucun)"
    missing_items = list(missing_fields)
    if missing_intent:
        missing_items.append("intent")

    missing_str = ", ".join(missing_items)

    lang_names = {"fr": "Français", "fon": "Fon", "yor": "Yoruba"}
    lang_label = lang_names.get(language, language)

    return f"""Tu es un assistant d'analyse linguistique pour la langue {lang_label}.

Texte transcrit :
"{raw_text}"

Informations déjà identifiées avec certitude (NE PAS MODIFIER NI INVENTER) :
{resolved_str}

Champs manquants que tu dois impérativement essayer d'extraire depuis le texte :
[{missing_str}]

Règles impératives :
1. Extrais les informations UNIQUEMENT si elles sont explicitement présentes ou déductibles du texte.
2. Si une information est introuvable ou incertaine, laisse-la à null.
3. Ne modifie pas les informations déjà identifiées.
4. Réponds strictement selon le schéma demandé.
"""


def extract_missing_fields_with_llm(
    raw_text: str,
    language: str,
    missing_fields: list[str],
    resolved_fields: dict[str, FieldMatch],
    missing_intent: bool = False,
    config: Optional[ComprehensionConfig] = None,
    client_override: Optional[instructor.Instructor] = None,
) -> tuple[dict[str, FieldMatch], Optional[FieldMatch]]:
    """
    Sollicite le LLM pour tenter de compléter uniquement les champs manquants.

    Retourne :
      - un dictionnaire de FieldMatch complétés (avec source=LLM ou source=MISSING)
      - un FieldMatch d'intention si elle était manquante et a été résolue, sinon None
    """
    if not missing_fields and not missing_intent:
        return {}, None

    cfg = config or ComprehensionConfig.from_env()
    client = client_override or get_instructor_client(cfg)

    # Si le LLM n'est pas disponible ou désactivé, retourner les champs comme toujours manquants
    if client is None:
        fallback_entities = {
            f: FieldMatch(field_name=f, value=None, source=FieldSource.MISSING)
            for f in missing_fields
        }
        return fallback_entities, None

    prompt = build_targeted_prompt(
        raw_text=raw_text,
        language=language,
        missing_fields=missing_fields,
        resolved_fields=resolved_fields,
        missing_intent=missing_intent,
    )

    try:
        response: LLMExtractionPayload = client.chat.completions.create(
            model=cfg.llm_model_name,
            response_model=LLMExtractionPayload,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        completed_entities: dict[str, FieldMatch] = {}
        for f in missing_fields:
            val = response.entities.get(f) if response.entities else None
            if val is not None and str(val).strip():
                completed_entities[f] = FieldMatch(
                    field_name=f,
                    value=str(val).strip(),
                    canonical_value=None,
                    score=None,
                    source=FieldSource.LLM,
                )
            else:
                completed_entities[f] = FieldMatch(
                    field_name=f,
                    value=None,
                    canonical_value=None,
                    score=None,
                    source=FieldSource.MISSING,
                )

        completed_intent = None
        if missing_intent and response.intent and str(response.intent).strip():
            completed_intent = FieldMatch(
                field_name="intent",
                value=str(response.intent).strip(),
                canonical_value=None,
                score=None,
                source=FieldSource.LLM,
            )

        return completed_entities, completed_intent

    except Exception as exc:
        logger.warning("Erreur lors de l'appel LLM Instructor : %s. Conservation des données existantes.", exc)
        fallback_entities = {
            f: FieldMatch(field_name=f, value=None, source=FieldSource.MISSING)
            for f in missing_fields
        }
        return fallback_entities, None
