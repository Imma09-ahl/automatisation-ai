"""
Rapprochement flou entre la transcription et un dictionnaire métier.

Utilise RapidFuzz pour comparer les tokens extraits de la transcription
avec les entrées du dictionnaire de référence.

Stratégie :
  - Seuil de similarité configurable via FUZZY_MATCH_THRESHOLD (défaut : 80).
  - Pour chaque champ métier attendu, on cherche la meilleure correspondance.
  - Les champs non résolus (score < seuil) sont retournés séparément
    pour être traités par le LLM.
"""

import os
import json
from pathlib import Path
from rapidfuzz import process, fuzz
from app.schemas import FieldMatch, FieldSource

# Seuil de similarité configurable
THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", "80"))


# ------------------------------------------------------------------ #
#  Chargement du dictionnaire
# ------------------------------------------------------------------ #

_dictionary: dict[str, list[str]] = {}


def load_dictionary(path: str | Path) -> dict[str, list[str]]:
    """
    Charge un dictionnaire métier depuis un fichier JSON.

    Format attendu :
    {
        "nom_champ": ["valeur1", "valeur2", ...],
        "autre_champ": ["valeurA", "valeurB", ...]
    }

    Paramètres
    ----------
    path : str | Path
        Chemin vers le fichier JSON du dictionnaire.

    Retourne
    --------
    dict[str, list[str]]
        Dictionnaire indexé par champ métier.
    """
    global _dictionary
    with open(path, "r", encoding="utf-8") as f:
        _dictionary = json.load(f)
    return _dictionary


def get_dictionary() -> dict[str, list[str]]:
    """Retourne le dictionnaire actuellement chargé."""
    return _dictionary


# ------------------------------------------------------------------ #
#  Rapprochement
# ------------------------------------------------------------------ #

def match_field(query: str, candidates: list[str]) -> tuple[str | None, float]:
    """
    Trouve la meilleure correspondance floue pour une valeur donnée.

    Paramètres
    ----------
    query : str
        Valeur issue de la transcription à comparer.
    candidates : list[str]
        Liste des valeurs possibles du dictionnaire.

    Retourne
    --------
    tuple[str | None, float]
        (meilleure correspondance, score) ou (None, 0.0) si aucun seuil atteint.
    """
    if not candidates:
        return None, 0.0

    result = process.extractOne(query, candidates, scorer=fuzz.WRatio)
    if result is None:
        return None, 0.0

    best_match, score, _ = result
    if score >= THRESHOLD:
        return best_match, score
    return None, score


def resolve_fields(
    transcription_text: str,
    expected_fields: list[str],
    dictionary: dict[str, list[str]] | None = None,
) -> tuple[dict[str, FieldMatch], list[str]]:
    """
    Tente de résoudre chaque champ métier attendu depuis la transcription.

    Paramètres
    ----------
    transcription_text : str
        Texte brut de la transcription.
    expected_fields : list[str]
        Liste des noms de champs à résoudre.
    dictionary : dict[str, list[str]], optionnel
        Dictionnaire à utiliser (utilise le dictionnaire global si None).

    Retourne
    --------
    tuple[dict[str, FieldMatch], list[str]]
        (champs résolus avec métadonnées, liste des champs manquants)
    """
    dico = dictionary or _dictionary
    tokens = transcription_text.lower().split()

    resolved: dict[str, FieldMatch] = {}
    missing: list[str] = []

    for field in expected_fields:
        candidates = dico.get(field, [])
        best_value = None
        best_score = 0.0

        # On teste chaque token contre les candidats du champ
        for token in tokens:
            value, score = match_field(token, candidates)
            if score > best_score:
                best_value = value
                best_score = score

        if best_value is not None:
            resolved[field] = FieldMatch(
                value=best_value,
                score=round(best_score, 2),
                source=FieldSource.DICTIONARY,
            )
        else:
            resolved[field] = FieldMatch(source=FieldSource.MISSING)
            missing.append(field)

    return resolved, missing
