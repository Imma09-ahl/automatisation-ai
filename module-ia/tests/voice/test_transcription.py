"""
Tests unitaires et d'intégration de la transcription trilingue (Français, Fon, Yoruba).

Couvre les 10 exigences du cahier des charges :
  1. Transcription française (faster-whisper)
  2. Transcription en fon (MMS-1b-all adaptateur fon)
  3. Transcription en yoruba (MMS-1b-all adaptateur yor, codes 'yo' et 'yor')
  4. Sélection automatique du bon moteur selon la langue
  5. Changement d'adaptateur MMS (fon <-> yor)
  6. Gestion de plusieurs segments audio
  7. Calcul des horodatages globaux avec décalage temporel
  8. Génération du fichier transcription.json unifié
  9. Gestion des erreurs (fichiers, métadonnées, modèles, transcription vide)
  10. Validation et normalisation des codes linguistiques
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from app.audio_errors import (
    AudioFileNotFoundError,
    EmptyTranscriptionError,
    MetadataError,
    MMSAdapterError,
    MMSModelLoadError,
    UnsupportedLanguageError,
    WhisperModelLoadError,
)
from app.mms_engine import reset_mms_engine, set_mms_adapter
from app.transcription import (
    ENGINE_MAP,
    TRANSCRIPTION_FILENAME,
    compute_global_timestamps,
    normalize_language_code,
    transcribe_segments_dir,
)


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_segments_dir(tmp_path: Path) -> Path:
    """Crée un dossier de segments simulé avec metadata.json et deux faux fichiers WAV."""
    seg_dir = tmp_path / "mock_session"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Création des faux fichiers WAV
    (seg_dir / "segment_0001.wav").write_bytes(b"RIFF" + b"\x00" * 80)
    (seg_dir / "segment_0002.wav").write_bytes(b"RIFF" + b"\x00" * 80)

    metadata = {
        "source": "conversation.wav",
        "duration": 12.0,
        "segments_count": 2,
        "sample_rate": 16000,
        "segments": [
            {
                "index": 1,
                "start": 1.0,
                "end": 4.5,
                "duration": 3.5,
                "file": "segment_0001.wav",
            },
            {
                "index": 2,
                "start": 6.0,
                "end": 10.0,
                "duration": 4.0,
                "file": "segment_0002.wav",
            },
        ],
    }
    (seg_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return seg_dir


# ------------------------------------------------------------------ #
#  1. Transcription française (faster-whisper)
# ------------------------------------------------------------------ #

def test_transcription_french_engine(mock_segments_dir: Path):
    with mock.patch("app.transcription.transcribe_french_segment") as mock_whisper:
        mock_whisper.side_effect = [
            [{"start": 0.2, "end": 3.0, "text": "Bonjour tout le monde."}],
            [{"start": 0.1, "end": 3.5, "text": "Comment allez-vous ?"}],
        ]

        result = transcribe_segments_dir(mock_segments_dir, language="fr")

        assert result["language"] == "fr"
        assert result["engine"] == "faster-whisper"
        assert result["text"] == "Bonjour tout le monde. Comment allez-vous ?"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Bonjour tout le monde."
        assert result["segments"][1]["text"] == "Comment allez-vous ?"


# ------------------------------------------------------------------ #
#  2. Transcription en Fon (MMS-1b-all adaptateur fon)
# ------------------------------------------------------------------ #

def test_transcription_fon_engine(mock_segments_dir: Path):
    with mock.patch("app.transcription.transcribe_mms_segment") as mock_mms:
        mock_mms.side_effect = [
            [{"start": 0.0, "end": 3.5, "text": "Afon ganjii a."}],
            [{"start": 0.0, "end": 4.0, "text": "Kudo zanze."}],
        ]

        result = transcribe_segments_dir(mock_segments_dir, language="fon")

        assert result["language"] == "fon"
        assert result["engine"] == "mms-1b-all"
        assert result["model"] == "facebook/mms-1b-all"
        assert result["text"] == "Afon ganjii a. Kudo zanze."
        assert mock_mms.call_count == 2
        # Vérifie que l'adaptateur fon a été demandé
        assert mock_mms.call_args_list[0].kwargs["adapter_lang"] == "fon"


# ------------------------------------------------------------------ #
#  3. Transcription en Yoruba (codes 'yo' et 'yor')
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("lang_input", ["yo", "yor", "YO", "YOR"])
def test_transcription_yoruba_engine(mock_segments_dir: Path, lang_input: str):
    with mock.patch("app.transcription.transcribe_mms_segment") as mock_mms:
        mock_mms.side_effect = [
            [{"start": 0.0, "end": 3.5, "text": "Bawo ni o se wa."}],
            [{"start": 0.0, "end": 4.0, "text": "E kaaro o."}],
        ]

        result = transcribe_segments_dir(mock_segments_dir, language=lang_input)

        assert result["language"] == "yor"
        assert result["engine"] == "mms-1b-all"
        assert result["text"] == "Bawo ni o se wa. E kaaro o."
        assert mock_mms.call_args_list[0].kwargs["adapter_lang"] == "yor"


# ------------------------------------------------------------------ #
#  4. Sélection du bon moteur selon la langue
# ------------------------------------------------------------------ #

def test_engine_selection_routing(mock_segments_dir: Path):
    with mock.patch("app.transcription.transcribe_french_segment", return_value=[{"text": "Salut"}]) as mock_whisper, \
         mock.patch("app.transcription.transcribe_mms_segment", return_value=[{"text": "E nle"}]) as mock_mms:

        # Appel FR -> faster-whisper seul doit être appelé
        transcribe_segments_dir(mock_segments_dir, language="fr")
        assert mock_whisper.called
        assert not mock_mms.called

        mock_whisper.reset_mock()
        mock_mms.reset_mock()

        # Appel FON -> MMS seul doit être appelé
        transcribe_segments_dir(mock_segments_dir, language="fon")
        assert not mock_whisper.called
        assert mock_mms.called


# ------------------------------------------------------------------ #
#  5. Changement d'adaptateur MMS (fon <-> yor)
# ------------------------------------------------------------------ #

def test_mms_adapter_switching_logic():
    reset_mms_engine()
    fake_model = mock.Mock()
    fake_processor = mock.Mock()

    # 1. Chargement de l'adaptateur fon
    set_mms_adapter(fake_model, fake_processor, "fon")
    fake_processor.tokenizer.set_target_lang.assert_called_with("fon")
    fake_model.load_adapter.assert_called_with("fon")

    fake_processor.tokenizer.set_target_lang.reset_mock()
    fake_model.load_adapter.reset_mock()

    # 2. Re-demande de fon -> ne doit pas recharger inutilement
    set_mms_adapter(fake_model, fake_processor, "fon")
    fake_processor.tokenizer.set_target_lang.assert_not_called()
    fake_model.load_adapter.assert_not_called()

    # 3. Changement vers yoruba -> doit charger yor
    set_mms_adapter(fake_model, fake_processor, "yor")
    fake_processor.tokenizer.set_target_lang.assert_called_with("yor")
    fake_model.load_adapter.assert_called_with("yor")

    # 4. Adaptateur non supporté
    with pytest.raises(MMSAdapterError, match="non pris en charge"):
        set_mms_adapter(fake_model, fake_processor, "esp")


# ------------------------------------------------------------------ #
#  6. Gestion de plusieurs segments audio
# ------------------------------------------------------------------ #

def test_multiple_audio_segments_handling(mock_segments_dir: Path):
    with mock.patch("app.transcription.transcribe_french_segment") as mock_whisper:
        mock_whisper.side_effect = [
            [{"start": 0.0, "end": 2.0, "text": "Premier segment."}],
            [{"start": 0.0, "end": 2.0, "text": "Deuxième segment."}],
        ]

        res = transcribe_segments_dir(mock_segments_dir, language="fr")
        assert res["segments_count"] == 2
        assert len(res["segments"]) == 2
        assert res["text"] == "Premier segment. Deuxième segment."


# ------------------------------------------------------------------ #
#  7. Calcul des horodatages globaux avec décalage
# ------------------------------------------------------------------ #

def test_global_timestamps_calculation():
    # Segment commençant à 5.0 s et finissant à 9.5 s dans le fichier original
    segment_offset = 5.0
    segment_end = 9.5
    local_pieces = [
        {"start": 0.5, "end": 2.1, "text": "Bonjour"},
        {"start": 2.5, "end": 4.0, "text": "monde"},
    ]

    global_pieces = compute_global_timestamps(segment_offset, local_pieces, segment_end)

    assert len(global_pieces) == 2
    assert global_pieces[0]["start"] == 5.5
    assert global_pieces[0]["end"] == 7.1
    assert global_pieces[0]["text"] == "Bonjour"

    assert global_pieces[1]["start"] == 7.5
    assert global_pieces[1]["end"] == 9.0
    assert global_pieces[1]["text"] == "monde"


# ------------------------------------------------------------------ #
#  8. Génération du JSON et structure identique pour les 3 langues
# ------------------------------------------------------------------ #

def test_json_generation_schema_uniformity(mock_segments_dir: Path):
    common_keys = {"source", "language", "engine", "model", "text", "segments_count", "segments"}
    seg_keys = {"index", "start", "end", "duration", "file", "text"}

    # Pour FR
    with mock.patch("app.transcription.transcribe_french_segment", return_value=[{"text": "Test FR"}]):
        res_fr = transcribe_segments_dir(mock_segments_dir, language="fr")
        assert common_keys <= res_fr.keys()
        assert seg_keys <= res_fr["segments"][0].keys()

    # Pour FON
    with mock.patch("app.transcription.transcribe_mms_segment", return_value=[{"text": "Test FON"}]):
        res_fon = transcribe_segments_dir(mock_segments_dir, language="fon")
        assert common_keys <= res_fon.keys()
        assert seg_keys <= res_fon["segments"][0].keys()

    # Pour YOR
    with mock.patch("app.transcription.transcribe_mms_segment", return_value=[{"text": "Test YOR"}]):
        res_yor = transcribe_segments_dir(mock_segments_dir, language="yor")
        assert common_keys <= res_yor.keys()
        assert seg_keys <= res_yor["segments"][0].keys()

    # Vérification de l'écriture physique de transcription.json
    json_file = mock_segments_dir / TRANSCRIPTION_FILENAME
    assert json_file.is_file()
    saved = json.loads(json_file.read_text(encoding="utf-8"))
    assert saved["language"] == "yor"
    assert saved["text"] == "Test YOR Test YOR"


# ------------------------------------------------------------------ #
#  9. Gestion des erreurs de fichiers, métadonnées et modèles
# ------------------------------------------------------------------ #

def test_error_missing_segments_directory(tmp_path: Path):
    with pytest.raises(AudioFileNotFoundError, match="introuvable"):
        transcribe_segments_dir(tmp_path / "inexistant", language="fr")


def test_error_missing_metadata_json(tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(MetadataError, match="metadata.json"):
        transcribe_segments_dir(empty_dir, language="fr")


def test_error_missing_segment_wav_file(mock_segments_dir: Path):
    # Supprimer un des fichiers audio de segment
    (mock_segments_dir / "segment_0001.wav").unlink()
    with pytest.raises(MetadataError, match="manquant"):
        transcribe_segments_dir(mock_segments_dir, language="fr")


def test_error_empty_transcription_rejected(mock_segments_dir: Path):
    with mock.patch("app.transcription.transcribe_french_segment", return_value=[]):
        with pytest.raises(EmptyTranscriptionError, match="Aucun texte"):
            transcribe_segments_dir(mock_segments_dir, language="fr", allow_empty=False)


def test_error_model_load_failure(mock_segments_dir: Path):
    with mock.patch("app.faster_whisper_engine.WhisperModel", side_effect=RuntimeError("Erreur mémoire")), \
         pytest.raises(WhisperModelLoadError, match="Impossible de charger le modèle"):
        transcribe_segments_dir(mock_segments_dir, language="fr")


# ------------------------------------------------------------------ #
#  10. Validation des codes linguistiques
# ------------------------------------------------------------------ #

def test_language_code_validation_and_normalization():
    assert normalize_language_code("fr") == "fr"
    assert normalize_language_code("FR") == "fr"
    assert normalize_language_code("fon") == "fon"
    assert normalize_language_code("FON") == "fon"
    assert normalize_language_code("yo") == "yor"
    assert normalize_language_code("yor") == "yor"
    assert normalize_language_code("  Yor  ") == "yor"

    with pytest.raises(UnsupportedLanguageError, match="non prise en charge"):
        normalize_language_code("en")

    with pytest.raises(UnsupportedLanguageError, match="non prise en charge"):
        normalize_language_code("es")

    with pytest.raises(UnsupportedLanguageError, match="doit être une chaîne"):
        normalize_language_code(123)  # type: ignore
