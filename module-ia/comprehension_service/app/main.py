"""
Point d'entrée de l'API web du Comprehension Service (FastAPI).

Routes exposées :
  GET  /health                  → vérification de l'état du service
  POST /comprehension/analyze   → analyse sémantique et structuration (FR, Fon, Yoruba)
  POST /understand              → alias direct de /comprehension/analyze
  GET  /dictionaries/{language} → consultation du dictionnaire chargé en mémoire
"""

from __future__ import annotations

import os
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from fastapi import FastAPI, HTTPException, Path as FastPath
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

import sys
from pathlib import Path

_comp_service_dir = Path(__file__).resolve().parent.parent
if str(_comp_service_dir) not in sys.path:
    sys.path.insert(0, str(_comp_service_dir))

from app.comprehension import analyze_transcription
from app.config import ComprehensionConfig
from app.dictionary_matcher import load_language_dictionary
from app.schemas import ComprehensionOutput, TranscriptionInput


# --- Handlers d'Endpoints ---

async def health() -> dict[str, Any]:
    """Vérifie que le service compréhension est opérationnel."""
    return {
        "status": "ok",
        "service": "comprehension",
        "supported_languages": ["fr", "fon", "yor"],
        "version": "0.2.0",
    }


async def analyze_endpoint(payload: TranscriptionInput) -> ComprehensionOutput:
    """
    Analyse une transcription et retourne les données structurées.
    Pipeline : Normalisation NFC -> RapidFuzz (Dictionnaire) -> Pydantic -> LLM conditionnel (Instructor).
    """
    try:
        return analyze_transcription(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur interne d'analyse : {exc}")


async def get_dictionary(language: str) -> dict[str, Any]:
    """
    Consulte le dictionnaire de termes actuellement configuré pour une langue.
    """
    lang = language.strip().lower()
    if lang in ("fr", "fra"):
        canon_lang = "fr"
    elif lang == "fon":
        canon_lang = "fon"
    elif lang in ("yo", "yor"):
        canon_lang = "yor"
    else:
        raise HTTPException(status_code=400, detail=f"Langue non supportée : '{language}'. Formats valides : fr, fon, yor.")

    try:
        dict_data = load_language_dictionary(canon_lang)
        return dict_data
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de chargement du dictionnaire : {exc}")


# --- Enregistrement de l'Application FastAPI ---

if HAS_FASTAPI:
    app = FastAPI(
        title="Comprehension Service",
        description="Service de compréhension et structuration linguistique (FR, Fon, Yoruba)",
        version="0.2.0",
    )

    app.get("/health")(health)
    app.post("/comprehension/analyze", response_model=ComprehensionOutput)(analyze_endpoint)
    app.post("/understand", response_model=ComprehensionOutput, include_in_schema=False)(analyze_endpoint)
    app.get("/dictionaries/{language}")(get_dictionary)
else:
    app = None
