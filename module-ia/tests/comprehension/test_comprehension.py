"""
Suite de tests automatisés pour le Comprehension Service.

Couvre l'ensemble des 15 exigences spécifiées :
  1. Réception d'une transcription française
  2. Réception d'une transcription en fon
  3. Réception d'une transcription en yoruba
  4. Normalisation des caractères Unicode et préservation des diacritiques
  5. Rapprochement avec RapidFuzz (ratio, partial_ratio, process.extract)
  6. Gestion des correspondances exactes
  7. Gestion des correspondances approximatives
  8. Gestion des termes inconnus
  9. Validation des modèles Pydantic
  10. Identification rigoureuse des champs manquants
  11. Absence stricte d'appel LLM lorsque tous les champs sont présents
  12. Appel LLM uniquement lorsqu'un champ nécessaire est manquant
  13. Gestion gracieuse d'un LLM indisponible (pas de crash, conservation des acquis)
  14. Gestion d'une réponse LLM invalide ou d'une exception réseau
  15. Conservation absolue des informations déjà résolues par le dictionnaire
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = REPO_ROOT / "comprehension_service"

if "app" in sys.modules:
    app_file = getattr(sys.modules["app"], "__file__", "")
    if app_file and "comprehension_service" not in app_file:
        for k in list(sys.modules.keys()):
            if k == "app" or k.startswith("app."):
                del sys.modules[k]

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from app.comprehension import analyze_transcription
from app.config import ComprehensionConfig
from app.dictionary_matcher import (
    calculate_match_score,
    clear_dictionaries_cache,
    find_best_candidate,
    load_language_dictionary,
    set_custom_dictionary,
)
from app.normalization import clean_token, normalize_text, tokenize_words
from app.schemas import (
    ComprehensionOutput,
    FieldMatch,
    FieldSource,
    LLMExtractionPayload,
    TranscriptionInput,
)


@pytest.fixture(autouse=True)
def clean_cache():
    clear_dictionaries_cache()
    yield
    clear_dictionaries_cache()


# ------------------------------------------------------------------ #
#  1, 2, 3. Réception de transcriptions en FR, FON et YORUBA
# ------------------------------------------------------------------ #

def test_french_transcription():
    text = "Je veux envoyer de l'argent à maman en fcfa"
    res = analyze_transcription(text, language="fr")

    assert res.language == "fr"
    assert res.raw_text == text
    assert res.intent is not None
    assert res.intent.value == "transfert_argent"
    assert res.entities["destinataire"].value == "maman"
    assert res.entities["devise"].value == "fcfa"
    assert res.is_complete is True
    assert res.llm_used is False


def test_fon_transcription():
    text = "Akwɛ́ sɛ́dó nɔ̂ en fcfa"
    res = analyze_transcription(text, language="fon")

    assert res.language == "fon"
    assert res.intent is not None
    assert res.intent.value == "transfert_argent"
    assert res.entities["destinataire"].value == "nɔ̂"
    assert res.entities["devise"].value == "fcfa"
    assert res.is_complete is True
    assert res.llm_used is False


def test_yoruba_transcription():
    text = "Mo fe fi owo ranse si iya mi ni naira"
    res = analyze_transcription(text, language="yor")

    assert res.language == "yor"
    assert res.intent is not None
    assert res.intent.value == "transfert_argent"
    assert res.entities["destinataire"].value == "iya mi"
    assert res.entities["devise"].value == "naira"
    assert res.is_complete is True
    assert res.llm_used is False


# ------------------------------------------------------------------ #
#  4. Normalisation Unicode et conservation des diacritiques
# ------------------------------------------------------------------ #

def test_unicode_normalization_preserves_diacritics():
    # Caractères spécifiques fon (ɖ, ɛ, ɔ, tons) et yoruba (subdots ẹ, ọ, ṣ)
    fon_phrase = "Kúdò   azɔ̌  ɖɔ :  É NYƆ́  "
    norm_fon = normalize_text(fon_phrase)
    assert norm_fon == "kúdò azɔ̌ ɖɔ : é nyɔ́"
    assert "ɖ" in norm_fon
    assert "ɔ̌" in norm_fon

    yor_phrase = "Ẹ KÚ  àárọ̀   o, gbogbo  nǹkan?"
    norm_yor = normalize_text(yor_phrase)
    assert norm_yor == "ẹ kú àárọ̀ o, gbogbo nǹkan?"
    assert "ẹ" in norm_yor
    assert "àárọ̀" in norm_yor

    # Tokens nettoyés
    tokens = tokenize_words("Akwɛ́, sɛ́dó! nɔ̂...")
    assert tokens == ["akwɛ́", "sɛ́dó", "nɔ̂"]


# ------------------------------------------------------------------ #
#  5, 6, 7, 8. Rapprochement RapidFuzz (exact, approximatif, inconnu)
# ------------------------------------------------------------------ #

def test_rapidfuzz_exact_and_approximate_matching():
    # Correspondance exacte
    assert calculate_match_score("transfert", "transfert") == 100.0

    # Correspondance avec casse et accents conservés
    assert calculate_match_score("AKWƐ́", "akwɛ́") == 100.0

    # Correspondance approximative (faute de frappe légère ou transcription)
    score_approx = calculate_match_score("transfere", "transfert")
    assert 75.0 <= score_approx < 100.0

    # Recherche avec candidats
    candidates = ["consulter_solde", "transfert_argent", "recharge_credit"]
    best, score = find_best_candidate("transfere argent", candidates, threshold=70.0)
    assert best == "transfert_argent"
    assert score >= 70.0


def test_unknown_terms_are_marked_missing():
    # Requête contenant des termes totalement hors dictionnaire
    text = "Hier j'ai mangé des bananes et une pomme"
    res = analyze_transcription(
        text,
        language="fr",
        expected_fields=["intent", "destinataire", "devise"],
        config=ComprehensionConfig(llm_enabled=False),
    )

    assert res.intent is None or res.intent.source == FieldSource.MISSING
    assert res.entities["destinataire"].source == FieldSource.MISSING
    assert res.entities["devise"].source == FieldSource.MISSING
    assert "destinataire" in res.missing_fields
    assert "devise" in res.missing_fields
    assert res.is_complete is False


# ------------------------------------------------------------------ #
#  9. Validation des modèles Pydantic
# ------------------------------------------------------------------ #

def test_pydantic_validation():
    # Texte vide rejeté
    with pytest.raises(ValidationError):
        TranscriptionInput(text="   ", language="fr")

    # Schéma ComprehensionOutput valide
    output = ComprehensionOutput(
        language="fr",
        raw_text="Test",
        normalized_text="test",
        intent=FieldMatch(field_name="intent", value="test", source=FieldSource.DICTIONARY),
        entities={},
        available_fields=["intent"],
        missing_fields=[],
        llm_used=False,
        is_complete=True,
    )
    assert output.is_complete is True


# ------------------------------------------------------------------ #
#  10. Identification des champs disponibles et manquants
# ------------------------------------------------------------------ #

def test_identification_of_missing_fields():
    # Seulement l'intention et la devise, sans destinataire
    text = "Je veux voir mon solde en fcfa"
    res = analyze_transcription(
        text,
        language="fr",
        expected_fields=["intent", "destinataire", "devise"],
        config=ComprehensionConfig(llm_enabled=False),
    )

    assert "intent" in res.available_fields
    assert "devise" in res.available_fields
    assert "destinataire" in res.missing_fields
    assert res.is_complete is False


# ------------------------------------------------------------------ #
#  11. RÈGLE CRITIQUE : AUCUN APPEL LLM LORSQUE TOUT EST PRÉSENT
# ------------------------------------------------------------------ #

def test_no_llm_called_when_all_fields_present():
    text = "Transfert pour maman en fcfa"
    mock_client = mock.Mock()

    res = analyze_transcription(
        text,
        language="fr",
        expected_fields=["intent", "destinataire", "devise"],
        client_override=mock_client,
    )

    # Vérification formelle : le client LLM n'a JAMAIS été appelé
    assert mock_client.chat.completions.create.call_count == 0
    assert res.llm_used is False
    assert res.is_complete is True
    assert res.entities["destinataire"].source == FieldSource.DICTIONARY
    assert res.entities["devise"].source == FieldSource.DICTIONARY


# ------------------------------------------------------------------ #
#  12. Appel LLM UNIQUEMENT pour les champs manquants
# ------------------------------------------------------------------ #

def test_llm_called_only_for_missing_fields():
    # Texte où le destinataire n'est pas un terme du dictionnaire ('Kofi' au lieu de 'maman')
    text = "Je veux envoyer de l'argent à Kofi en fcfa"

    mock_client = mock.Mock()
    # Réponse structurée Instructor mockée
    mock_client.chat.completions.create.return_value = LLMExtractionPayload(
        intent=None,
        entities={"destinataire": "Kofi"},
    )

    res = analyze_transcription(
        text,
        language="fr",
        expected_fields=["intent", "destinataire", "devise"],
        client_override=mock_client,
    )

    # Le LLM a été appelé car 'destinataire' manquait
    assert mock_client.chat.completions.create.call_count == 1
    assert res.llm_used is True

    # Vérification du prompt envoyé : seul 'destinataire' devait être demandé
    call_args = mock_client.chat.completions.create.call_args
    prompt_sent = call_args.kwargs["messages"][0]["content"]
    assert "destinataire" in prompt_sent
    assert "transfert_argent" in prompt_sent  # présent comme champ déjà identifié

    # Vérification des sources
    assert res.intent.source == FieldSource.DICTIONARY
    assert res.entities["devise"].source == FieldSource.DICTIONARY
    assert res.entities["destinataire"].source == FieldSource.LLM
    assert res.entities["destinataire"].value == "Kofi"
    assert res.is_complete is True


# ------------------------------------------------------------------ #
#  13. Gestion d'un LLM indisponible (Fallback propre)
# ------------------------------------------------------------------ #

def test_llm_unavailable_graceful_fallback():
    # Clé absente ou LLM désactivé
    cfg = ComprehensionConfig(llm_enabled=False, llm_api_key=None)

    text = "Envoyer 5000 à quelqu'un"
    res = analyze_transcription(
        text,
        language="fr",
        expected_fields=["intent", "destinataire"],
        config=cfg,
    )

    # Le service ne plante pas et retourne les informations du dictionnaire
    assert res.llm_used is False
    assert res.intent.value == "transfert_argent"
    assert res.intent.source == FieldSource.DICTIONARY
    assert res.entities["destinataire"].source == FieldSource.MISSING
    assert "destinataire" in res.missing_fields


# ------------------------------------------------------------------ #
#  14. Gestion d'une erreur ou réponse LLM invalide
# ------------------------------------------------------------------ #

def test_llm_error_recovery():
    mock_client = mock.Mock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API Timeout")

    text = "Envoyer des fonds à mon oncle"
    res = analyze_transcription(
        text,
        language="fr",
        expected_fields=["intent", "destinataire"],
        client_override=mock_client,
    )

    # L'erreur est gérée gracieusement, les données dictionnaire sont sauvées
    assert res.intent.value == "transfert_argent"
    assert res.intent.source == FieldSource.DICTIONARY
    assert res.entities["destinataire"].source == FieldSource.MISSING


# ------------------------------------------------------------------ #
#  15. Conservation absolue des informations déjà trouvées
# ------------------------------------------------------------------ #

def test_preservation_of_dictionary_fields_when_llm_responds():
    text = "Virement pour mon collègue en euros"

    mock_client = mock.Mock()
    # Le LLM tente de renvoyer une devise différente
    mock_client.chat.completions.create.return_value = LLMExtractionPayload(
        intent="autre_intention_inventee",
        entities={"destinataire": "collègue", "devise": "dollars_inventes"},
    )

    res = analyze_transcription(
        text,
        language="fr",
        expected_fields=["intent", "destinataire", "devise"],
        client_override=mock_client,
    )

    # L'intention et la devise trouvées par le dictionnaire ne sont PAS écrasées
    assert res.intent.value == "transfert_argent"
    assert res.intent.source == FieldSource.DICTIONARY
    assert res.entities["devise"].value == "euros"
    assert res.entities["devise"].source == FieldSource.DICTIONARY

    # Seul le champ manquant 'destinataire' est complété par le LLM
    assert res.entities["destinataire"].value == "collègue"
    assert res.entities["destinataire"].source == FieldSource.LLM
