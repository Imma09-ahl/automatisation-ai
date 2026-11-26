"""
Module principal du Comprehension Service.

Orchestre le pipeline de compréhension :
  1. Réception et validation de la transcription (Français, Fon, Yoruba).
  2. Normalisation du texte préservant les caractères Unicode et signes diacritiques.
  3. Rapprochement lexical flou avec le dictionnaire métier via RapidFuzz.
  4. Détection des champs disponibles et des champs manquants.
  5. RÈGLE CRITIQUE : Appel conditionnel au LLM via Instructor UNIQUEMENT
     si un ou plusieurs champs nécessaires sont manquants.
  6. Structuration et validation du résultat final sous forme de schéma Pydantic ComprehensionOutput.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union

_service_dir = Path(__file__).resolve().parent.parent
if str(_service_dir) not in sys.path:
    sys.path.insert(0, str(_service_dir))

from app.config import ComprehensionConfig
from app.dictionary_matcher import DictionaryMatcher, load_language_dictionary
from app.llm_service import extract_missing_fields_with_llm, get_instructor_client
from app.normalization import normalize_text
from app.schemas import (
    ComprehensionOutput,
    FieldMatch,
    FieldSource,
    TranscriptionInput,
)

PathLike = Union[str, Path]

# Champs attendus par défaut (peuvent être configurés ou surchargés)
DEFAULT_EXPECTED_FIELDS = ["intent", "destinataire", "devise"]


def normalize_language_code(language: str) -> str:
    """Normalise les codes de langue : 'fr'/'fra' -> 'fr', 'fon' -> 'fon', 'yo'/'yor' -> 'yor'."""
    lang = language.strip().lower()
    if lang in ("fr", "fra"):
        return "fr"
    elif lang == "fon":
        return "fon"
    elif lang in ("yo", "yor"):
        return "yor"
    return lang


def analyze_transcription(
    transcription_data: Union[TranscriptionInput, dict[str, Any], str],
    language: Optional[str] = None,
    expected_fields: Optional[list[str]] = None,
    config: Optional[ComprehensionConfig] = None,
    custom_dictionary: Optional[dict[str, Any]] = None,
    client_override: Any = None,
) -> ComprehensionOutput:
    """
    Analyse une transcription vocale pour en extraire l'intention et les entités.

    Paramètres
    ----------
    transcription_data : Objet TranscriptionInput, dictionnaire JSON (issu de transcription.json)
                         ou simple chaîne de texte.
    language : Code langue (si non présent dans transcription_data).
    expected_fields : Liste des champs/entités requis à extraire (défaut : DEFAULT_EXPECTED_FIELDS).
    config : Configuration du service (seuils, paramètres LLM).
    custom_dictionary : Dictionnaire optionnel pour forcer des entrées (utile pour les tests).
    client_override : Client Instructor mocké ou personnalisé.

    Retourne
    --------
    ComprehensionOutput
        Résultat complet, validé et conforme au schéma Pydantic unifié.
    """
    cfg = config or ComprehensionConfig.from_env()
    fields_to_check = expected_fields or list(DEFAULT_EXPECTED_FIELDS)

    # 1. Normalisation de l'entrée vers TranscriptionInput
    if isinstance(transcription_data, TranscriptionInput) or type(transcription_data).__name__ == "TranscriptionInput":
        inp = transcription_data
    elif isinstance(transcription_data, dict):
        raw_txt = transcription_data.get("text", "")
        lang = transcription_data.get("language") or language or "fr"
        source = transcription_data.get("source")
        segs = transcription_data.get("segments", [])
        inp = TranscriptionInput(text=raw_txt, language=lang, source=source, segments=segs)
    elif isinstance(transcription_data, str):
        inp = TranscriptionInput(text=transcription_data, language=language or "fr")
    else:
        raise ValueError(f"Format de transcription non supporté : {type(transcription_data)}")

    canonical_lang = normalize_language_code(inp.language)
    norm_text = normalize_text(inp.text)

    # 2. Chargement du dictionnaire linguistique approprié
    dictionary = custom_dictionary or load_language_dictionary(
        canonical_lang,
        dictionaries_dir=cfg.dictionaries_dir,
    )

    # 3. Rapprochement lexical avec RapidFuzz
    matcher = DictionaryMatcher(cfg)

    # Intention
    matched_intent: Optional[FieldMatch] = None
    if "intent" in fields_to_check:
        matched_intent = matcher.match_intent(inp.text, dictionary)

    # Entités
    entity_names = [f for f in fields_to_check if f != "intent"]
    resolved_entities, missing_entities = matcher.match_entities(
        inp.text,
        entity_names,
        dictionary,
    )

    # Évaluation des champs manquants
    missing_fields: list[str] = list(missing_entities)
    missing_intent = False
    if "intent" in fields_to_check and (matched_intent is None or not matched_intent.is_resolved):
        missing_intent = True
        missing_fields.insert(0, "intent")

    llm_used = False

    # 4. Décision conditionnelle : Appeler le LLM UNIQUEMENT si des champs sont manquants
    if missing_fields:
        # Vérifier si un client LLM est disponible
        client = client_override or get_instructor_client(cfg)
        if client is not None:
            # Inclure les entités résolues ainsi que l'intention résolue pour informer le LLM
            resolved_for_llm = dict(resolved_entities)
            if matched_intent and matched_intent.is_resolved:
                resolved_for_llm["intent"] = matched_intent

            # Appel ciblé pour les champs manquants uniquement
            llm_entities, llm_intent = extract_missing_fields_with_llm(
                raw_text=inp.text,
                language=canonical_lang,
                missing_fields=missing_entities,
                resolved_fields=resolved_for_llm,
                missing_intent=missing_intent,
                config=cfg,
                client_override=client,
            )
            llm_used = True

            # Mise à jour des entités avec les apports du LLM
            for k, match in llm_entities.items():
                if match.is_resolved:
                    resolved_entities[k] = match

            # Mise à jour de l'intention si elle a été complétée par le LLM
            if missing_intent and llm_intent and llm_intent.is_resolved:
                matched_intent = llm_intent

    # Recalcul des listes de champs disponibles et encore manquants
    available_fields: list[str] = []
    still_missing_fields: list[str] = []

    if matched_intent and matched_intent.is_resolved:
        available_fields.append("intent")
    elif "intent" in fields_to_check:
        still_missing_fields.append("intent")

    for f_name, match in resolved_entities.items():
        if match.is_resolved:
            available_fields.append(f_name)
        elif f_name in fields_to_check:
            still_missing_fields.append(f_name)

    is_complete = len(still_missing_fields) == 0

    return ComprehensionOutput(
        source=inp.source,
        language=canonical_lang,
        raw_text=inp.text,
        normalized_text=norm_text,
        intent=matched_intent,
        entities=resolved_entities,
        available_fields=available_fields,
        missing_fields=still_missing_fields,
        llm_used=llm_used,
        is_complete=is_complete,
    )


# ------------------------------------------------------------------ #
#  Ligne de commande
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehension Service : analyse et extraction structurée de transcriptions."
    )
    parser.add_argument("input", help="Texte à analyser OU chemin vers un fichier transcription.json.")
    parser.add_argument("-l", "--language", default="fr", help="Code langue ('fr', 'fon', 'yo'/'yor'). Défaut : fr.")
    parser.add_argument("--threshold", type=float, default=75.0, help="Seuil de similarité RapidFuzz (0 à 100).")
    parser.add_argument("--no-llm", action="store_true", help="Désactive formellement tout appel LLM.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    target = Path(args.input)

    cfg = ComprehensionConfig.from_env()
    if args.threshold:
        cfg = ComprehensionConfig(
            fuzzy_match_threshold=args.threshold,
            llm_enabled=not args.no_llm and cfg.llm_enabled,
            llm_api_key=cfg.llm_api_key,
        )

    if target.is_file() and target.suffix == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
        res = analyze_transcription(data, config=cfg)
    else:
        res = analyze_transcription(args.input, language=args.language, config=cfg)

    print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
