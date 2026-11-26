"""
Normalisation linguistique des textes pour le service de compréhension.

Préserve scrupuleusement les caractères Unicode et signes diacritiques
essentiels au Fon et au Yoruba (ɛ, ɔ, ɖ, sous-points ẹ, ọ, ṣ, accents tonaux),
tout en harmonisant la casse, les espaces et la forme canonique Unicode (NFC).
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_REGEX = re.compile(r"\s+")
_PUNCTUATION_EDGE_REGEX = re.compile(r"^[\s.,!?;:\"'«»—\-()]+|[\s.,!?;:\"'«»—\-()]+$")


def normalize_text(text: str) -> str:
    """
    Normalise une chaîne de texte :
      1. Normalisation Unicode NFC (canonical composition).
      2. Mise en minuscules (case folding) sans supprimer les diacritiques.
      3. Remplacement des espaces multiples par un espace unique.
      4. Suppression des espaces en début et fin de chaîne.
    """
    if not text:
        return ""

    # Normalisation Unicode NFC pour unifier les glyphes composés/décomposés
    nfc_text = unicodedata.normalize("NFC", text)

    # Minuscules (conserve les caractères comme é, è, ɛ, ɔ, ɖ, ẹ, ọ, etc.)
    lowered = nfc_text.lower()

    # Remplacement des espaces consécutifs et retours à la ligne
    collapsed = _WHITESPACE_REGEX.sub(" ", lowered).strip()

    return collapsed


def clean_token(token: str) -> str:
    """Nettoie un token individuel de sa ponctuation périphérique."""
    cleaned = normalize_text(token)
    return _PUNCTUATION_EDGE_REGEX.sub("", cleaned)


def tokenize_words(text: str) -> list[str]:
    """Découpe un texte en mots nettoyés tout en préservant l'ordre et les caractères diacritiques."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    words = normalized.split(" ")
    cleaned = [clean_token(w) for w in words]
    return [w for w in cleaned if w]
