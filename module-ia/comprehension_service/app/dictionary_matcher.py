"""
Rapprochement lexical avec dictionnaire métier utilisant RapidFuzz.

Implémente :
  - fuzz.ratio pour la similarité globale exacte/approximative.
  - fuzz.partial_ratio pour la détection de sous-chaînes ou expressions composées.
  - process.extract et process.extractOne pour la recherche des meilleurs candidats.
  - Prise en compte rigoureuse des caractères Unicode et signes diacritiques en Fon et Yoruba.
  - Gestion des dictionnaires par langue (FR, Fon, Yoruba) avec chargement dynamique.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from rapidfuzz import fuzz, process

from app.config import ComprehensionConfig
from app.normalization import clean_token, normalize_text, tokenize_words
from app.schemas import FieldMatch, FieldSource

# Cache des dictionnaires chargés en mémoire par langue
_loaded_dictionaries: dict[str, dict[str, Any]] = {}


def load_language_dictionary(
    language: str,
    dictionaries_dir: Optional[Path] = None,
    force_reload: bool = False,
) -> dict[str, Any]:
    """
    Charge le fichier JSON du dictionnaire pour une langue donnée ('fr', 'fon', 'yor').
    Si le dictionnaire n'est pas trouvé ou invalide, retourne une structure vide par défaut.
    """
    lang_key = language.strip().lower()
    if lang_key in ("yo", "yor"):
        lang_key = "yor"
    elif lang_key in ("fr", "fra"):
        lang_key = "fr"

    if not force_reload and lang_key in _loaded_dictionaries:
        return _loaded_dictionaries[lang_key]

    base_dir = dictionaries_dir or ComprehensionConfig.from_env().dictionaries_dir
    dict_file = base_dir / f"{lang_key}.json"

    if not dict_file.is_file():
        # Dictionnaire absent : structure générique vide
        empty_struct = {
            "language": lang_key,
            "intent": {},
            "entities": {},
        }
        _loaded_dictionaries[lang_key] = empty_struct
        return empty_struct

    try:
        content = json.loads(dict_file.read_text(encoding="utf-8"))
        _loaded_dictionaries[lang_key] = content
        return content
    except Exception:
        empty_struct = {"language": lang_key, "intent": {}, "entities": {}}
        _loaded_dictionaries[lang_key] = empty_struct
        return empty_struct


def set_custom_dictionary(language: str, data: dict[str, Any]) -> None:
    """Injecte un dictionnaire personnalisé en mémoire (utile pour les tests)."""
    lang_key = language.strip().lower()
    if lang_key in ("yo", "yor"):
        lang_key = "yor"
    elif lang_key in ("fr", "fra"):
        lang_key = "fr"
    _loaded_dictionaries[lang_key] = data


def clear_dictionaries_cache() -> None:
    """Vide le cache des dictionnaires."""
    _loaded_dictionaries.clear()


def calculate_match_score(query: str, candidate: str, **kwargs: Any) -> float:
    """
    Calcule le score de correspondance combiné entre une chaîne requête et un candidat :
    Combine fuzz.ratio (similarité globale), fuzz.token_sort_ratio et fuzz.partial_ratio sécurisé.
    Accepte **kwargs pour la compatibilité avec RapidFuzz process.extract (score_cutoff, etc.).
    """
    norm_query = normalize_text(query)
    norm_cand = normalize_text(candidate)

    if not norm_query or not norm_cand:
        return 0.0

    # Correspondance exacte directe après normalisation
    if norm_query == norm_cand:
        return 100.0

    ratio_score = fuzz.ratio(norm_query, norm_cand)
    token_score = fuzz.token_sort_ratio(norm_query, norm_cand)
    score = float(max(ratio_score, token_score))

    # partial_ratio sécurisé uniquement si les longueurs sont comparables
    # pour éviter que des mots courts (1-2 lettres) ne matchent des termes aléatoires
    len_q = len(norm_query)
    len_c = len(norm_cand)
    if len_q >= 3 and len_c >= 3:
        len_ratio = min(len_q, len_c) / max(len_q, len_c)
        if len_ratio >= 0.65:
            partial_score = fuzz.partial_ratio(norm_query, norm_cand)
            if partial_score >= 80.0:
                score = max(score, float(partial_score * 0.95))

    score_cutoff = kwargs.get("score_cutoff")
    if score_cutoff is not None and score < score_cutoff:
        return 0.0

    return score


def find_best_candidate(
    query: str,
    candidates: list[str],
    threshold: float = 75.0,
) -> tuple[Optional[str], float]:
    """
    Trouve le meilleur candidat parmi une liste à l'aide de process.extract / process.extractOne et calculate_match_score.
    Retourne (candidat, score) si score >= threshold, sinon (None, score_max).
    """
    if not query or not candidates:
        return None, 0.0

    norm_query = normalize_text(query)

    # Utilisation de process.extract pour évaluer les meilleurs candidats
    extracted = process.extract(
        norm_query,
        candidates,
        scorer=calculate_match_score,
        limit=5,
    )

    if extracted:
        top_cand, top_score, _ = extracted[0]
        best_score = round(float(top_score), 2)
        if best_score >= threshold:
            return top_cand, best_score

    return None, 0.0


class DictionaryMatcher:
    """
    Moteur de rapprochement lexical entre une transcription et un dictionnaire de concepts/entités.
    """

    def __init__(self, config: Optional[ComprehensionConfig] = None) -> None:
        self.config = config or ComprehensionConfig.from_env()

    def match_intent(
        self,
        text: str,
        dictionary: dict[str, Any],
    ) -> Optional[FieldMatch]:
        """
        Détecte l'intention principale en comparant le texte et ses n-grammes
        aux expressions associées à chaque intention du dictionnaire.
        Privilégie les expressions les plus longues et spécifiques en cas d'égalité de score.
        """
        intents_dict = dictionary.get("intent", {})
        if not intents_dict:
            return None

        normalized_text = normalize_text(text)
        words = tokenize_words(text)

        best_intent_key: Optional[str] = None
        best_score: float = 0.0
        best_specificity: int = 0

        for intent_name, expressions in intents_dict.items():
            if not isinstance(expressions, list):
                continue

            for expr in expressions:
                norm_expr = normalize_text(expr)
                if not norm_expr:
                    continue

                # 1. Correspondance exacte par inclusion de la locution dans le texte normalisé
                if norm_expr in normalized_text:
                    score = 100.0
                    specificity = len(norm_expr)
                    if (score > best_score) or (score == best_score and specificity > best_specificity):
                        best_score = score
                        best_specificity = specificity
                        best_intent_key = intent_name
                else:
                    # 2. Correspondance floue par n-grammes de taille similaire
                    n_words = len(norm_expr.split())
                    window_sizes = {n_words, n_words + 1, max(1, n_words - 1)}
                    for win in window_sizes:
                        for i in range(len(words) - win + 1):
                            ngram = " ".join(words[i : i + win])
                            s = calculate_match_score(ngram, norm_expr)
                            if s >= self.config.fuzzy_match_threshold:
                                specificity = len(norm_expr)
                                if (s > best_score) or (s == best_score and specificity > best_specificity):
                                    best_score = s
                                    best_specificity = specificity
                                    best_intent_key = intent_name

        if best_intent_key is not None and best_score >= self.config.fuzzy_match_threshold:
            return FieldMatch(
                field_name="intent",
                value=best_intent_key,
                canonical_value=best_intent_key,
                score=round(best_score, 2),
                source=FieldSource.DICTIONARY,
            )

        return None

    def match_entities(
        self,
        text: str,
        expected_fields: list[str],
        dictionary: dict[str, Any],
    ) -> tuple[dict[str, FieldMatch], list[str]]:
        """
        Recherche les entités attendues dans la transcription.
        Retourne :
          - entities: dictionnaire de FieldMatch pour chaque entité
          - missing: liste des champs attendus qui n'ont pas pu être résolus
        """
        entities_dict = dictionary.get("entities", {})
        words = tokenize_words(text)

        resolved_entities: dict[str, FieldMatch] = {}
        missing_fields: list[str] = []

        for field in expected_fields:
            if field == "intent":
                continue

            candidates = entities_dict.get(field, [])
            best_val: Optional[str] = None
            best_score: float = 0.0

            if candidates and isinstance(candidates, list):
                # Tester chaque mot et n-gramme contre les candidats
                for n in range(1, min(4, len(words) + 1)):
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i : i + n])
                        cand, score = find_best_candidate(
                            ngram,
                            candidates,
                            threshold=self.config.fuzzy_match_threshold,
                        )
                        if score > best_score:
                            best_score = score
                            best_val = cand

            if best_val is not None and best_score >= self.config.fuzzy_match_threshold:
                resolved_entities[field] = FieldMatch(
                    field_name=field,
                    value=best_val,
                    canonical_value=best_val,
                    score=round(best_score, 2),
                    source=FieldSource.DICTIONARY,
                )
            else:
                resolved_entities[field] = FieldMatch(
                    field_name=field,
                    value=None,
                    canonical_value=None,
                    score=round(best_score, 2) if best_score > 0 else None,
                    source=FieldSource.MISSING,
                )
                missing_fields.append(field)

        return resolved_entities, missing_fields
