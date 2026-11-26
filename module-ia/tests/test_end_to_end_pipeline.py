"""
Suite de tests de bout en bout (End-to-End) et interopérabilité des services.

Vérifie l'enchaînement complet du pipeline :
  1. Disponibilité des services (Health checks)
  2. Traitement d'entrées de transcription (FR, Fon, Yoruba)
  3. Transmission transparente de transcription.json vers le service compréhension
  4. Rapprochement dictionnaire RapidFuzz
  5. Appel conditionnel du LLM
  6. Validation Pydantic et génération du JSON final standardisé
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
COMP_DIR = REPO_ROOT / "comprehension_service"

if str(COMP_DIR) not in sys.path:
    sys.path.insert(0, str(COMP_DIR))

from app.comprehension import analyze_transcription, normalize_language_code
from app.config import ComprehensionConfig
from app.main import health as comp_health
from app.schemas import (
    ComprehensionOutput,
    FieldMatch,
    FieldSource,
    LLMExtractionPayload,
    TranscriptionInput,
)


# ------------------------------------------------------------------ #
#  1. Vérification de disponibilité du service compréhension (Health)
# ------------------------------------------------------------------ #

def test_comprehension_health_availability():
    """Vérifie que le service compréhension répond 'ok' sur son endpoint health."""
    c_res = asyncio.run(comp_health())
    assert c_res["status"] == "ok"
    assert c_res["service"] == "comprehension"
    assert "fr" in c_res["supported_languages"]
    assert "fon" in c_res["supported_languages"]
    assert "yor" in c_res["supported_languages"]


# ------------------------------------------------------------------ #
#  2. Cohérence des codes linguistiques
# ------------------------------------------------------------------ #

def test_language_codes_uniformity():
    """Vérifie la normalisation uniforme des codes de langue (fr, fon, yor)."""
    assert normalize_language_code("fr") == "fr"
    assert normalize_language_code("fra") == "fr"
    assert normalize_language_code("fon") == "fon"
    assert normalize_language_code("yo") == "yor"
    assert normalize_language_code("yor") == "yor"


# ------------------------------------------------------------------ #
#  3. Pipeline Bout en Bout : Français (Audio Transcrit -> Compréhension)
# ------------------------------------------------------------------ #

def test_end_to_end_french_pipeline_without_llm():
    """
    Simulation du flux complet en français :
    Le dictionnaire JSON produit par voice_service (transcription.json)
    est injecté directement dans analyze_transcription.
    Tous les champs sont résolus par le dictionnaire, zéro appel LLM.
    """
    # 1. Structure JSON exacte produite par voice_service/transcribe
    simulated_voice_output = {
        "text": "Je veux envoyer de l'argent à maman en fcfa",
        "language": "fr",
        "duration": 3.8,
        "segments": [{"id": 1, "start": 0.0, "end": 3.8, "text": "Je veux envoyer de l'argent à maman en fcfa"}],
        "source": "voice_service_transcribe",
    }

    mock_llm_client = mock.Mock()
    result = analyze_transcription(
        simulated_voice_output,
        expected_fields=["intent", "destinataire", "devise"],
        client_override=mock_llm_client,
    )

    # 2. Assertions sur le résultat final
    assert isinstance(result, ComprehensionOutput)
    assert result.source == "voice_service_transcribe"
    assert result.language == "fr"
    assert result.intent is not None
    assert result.intent.value == "transfert_argent"
    assert result.intent.source == FieldSource.DICTIONARY
    assert result.entities["destinataire"].value == "maman"
    assert result.entities["devise"].value == "fcfa"
    assert result.missing_fields == []
    assert result.llm_used is False
    assert result.is_complete is True
    # Zéro appel LLM garanti
    assert mock_llm_client.chat.completions.create.call_count == 0


# ------------------------------------------------------------------ #
#  4. Pipeline Bout en Bout : Fon avec préservation des diacritiques
# ------------------------------------------------------------------ #

def test_end_to_end_fon_pipeline():
    """
    Simulation du flux complet en fon avec caractères diacritiques et tons.
    """
    simulated_voice_output = {
        "text": "Akwɛ́ sɛ́dó nɔ̂ en fcfa",
        "language": "fon",
        "duration": 2.9,
        "segments": [{"id": 1, "start": 0.0, "end": 2.9, "text": "Akwɛ́ sɛ́dó nɔ̂ en fcfa"}],
    }

    result = analyze_transcription(
        simulated_voice_output,
        expected_fields=["intent", "destinataire", "devise"],
    )

    assert isinstance(result, ComprehensionOutput)
    assert result.language == "fon"
    assert result.intent.value == "transfert_argent"
    assert result.entities["destinataire"].value == "nɔ̂"
    assert result.entities["devise"].value == "fcfa"
    assert result.is_complete is True
    assert result.llm_used is False


# ------------------------------------------------------------------ #
#  5. Pipeline Bout en Bout : Yoruba
# ------------------------------------------------------------------ #

def test_end_to_end_yoruba_pipeline():
    """
    Simulation du flux complet en yoruba.
    """
    simulated_voice_output = {
        "text": "Mo fe fi owo ranse si iya mi ni naira",
        "language": "yor",
        "duration": 3.4,
    }

    result = analyze_transcription(
        simulated_voice_output,
        expected_fields=["intent", "destinataire", "devise"],
    )

    assert isinstance(result, ComprehensionOutput)
    assert result.language == "yor"
    assert result.intent.value == "transfert_argent"
    assert result.entities["destinataire"].value == "iya mi"
    assert result.entities["devise"].value == "naira"
    assert result.is_complete is True
    assert result.llm_used is False


# ------------------------------------------------------------------ #
#  6. Pipeline avec déclenchement conditionnel du LLM pour champ manquant
# ------------------------------------------------------------------ #

def test_end_to_end_pipeline_with_conditional_llm():
    """
    Vérifie qu'un champ inconnu du dictionnaire déclenche le LLM
    sans altérer les champs déjà résolus par le dictionnaire.
    """
    simulated_voice_output = {
        "text": "Je veux envoyer de l'argent à Kofi en fcfa",
        "language": "fr",
    }

    mock_llm_client = mock.Mock()
    mock_llm_client.chat.completions.create.return_value = LLMExtractionPayload(
        intent=None,
        entities={"destinataire": "Kofi"},
    )

    result = analyze_transcription(
        simulated_voice_output,
        expected_fields=["intent", "destinataire", "devise"],
        client_override=mock_llm_client,
    )

    assert result.llm_used is True
    assert mock_llm_client.chat.completions.create.call_count == 1
    assert result.intent.value == "transfert_argent"
    assert result.intent.source == FieldSource.DICTIONARY
    assert result.entities["destinataire"].value == "Kofi"
    assert result.entities["destinataire"].source == FieldSource.LLM
    assert result.entities["devise"].value == "fcfa"
    assert result.entities["devise"].source == FieldSource.DICTIONARY
    assert result.is_complete is True


# ------------------------------------------------------------------ #
#  7. Validation Pydantic et sérialisation JSON pour la plateforme
# ------------------------------------------------------------------ #

def test_pydantic_json_serialization_compatibility():
    """
    Vérifie que la sortie est sérialisable en JSON UTF-8 standard
    exploitable par n'importe quel backend de l'équipe (Node, Spring, etc.).
    """
    inp = TranscriptionInput(
        text="Akwɛ́ sɛ́dó nɔ̂ en fcfa",
        language="fon",
        expected_fields=["intent", "destinataire", "devise"],
    )
    result = analyze_transcription(inp)

    json_str = result.model_dump_json()
    assert isinstance(json_str, str)
    assert "Akwɛ́ sɛ́dó" in json_str or "akwɛ́ sɛ́dó" in json_str
    assert "transfert_argent" in json_str
    assert "dictionary" in json_str
