"""
Tests unitaires de la synthèse vocale (Text-to-Speech / MMS-TTS).

Couvre les 12 points du cahier des charges :
  1. Sélection du modèle français (facebook/mms-tts-fra)
  2. Sélection du modèle fon (facebook/mms-tts-fon)
  3. Sélection du modèle yoruba (facebook/mms-tts-yor)
  4. Génération d'un texte court
  5. Gestion des textes longs avec découpage et silences
  6. Sauvegarde correcte des fichiers WAV au format PCM 16 bits
  7. Génération des métadonnées JSON
  8. Conservation intégrale des caractères Unicode et signes diacritiques (fon et yoruba)
  9. Gestion des langues non prises en charge
  10. Gestion des erreurs de chargement de modèle
  11. Réutilisation du modèle en mémoire pour une même langue
  12. Gestion des ressources et déchargement de l'ancien modèle lors du changement de langue
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from app.audio_errors import (
    EmptyTextError,
    TTSGenerationError,
    TTSModelLoadError,
    UnsupportedLanguageError,
)
from app.speech_synthesis import (
    MMS_TTS_MODELS,
    TTSModelManager,
    normalize_tts_language,
    reset_tts_models,
    save_wav_pcm16,
    split_text_into_sentences,
    synthesize,
)


# ------------------------------------------------------------------ #
#  Fixtures et Mocks
# ------------------------------------------------------------------ #

class FakeVitsOutput:
    def __init__(self, waveform: torch.Tensor):
        self.waveform = waveform


def make_fake_vits_model(sample_rate: int = 16000):
    model = mock.Mock()
    model.config = mock.Mock()
    model.config.sampling_rate = sample_rate

    # Génère 0.5s d'onde sonore (8000 échantillons à 16kHz)
    wave_tensor = torch.sin(torch.linspace(0, 10, 8000)).unsqueeze(0)
    model.return_value = FakeVitsOutput(wave_tensor)
    return model


def make_fake_tokenizer():
    tokenizer = mock.Mock()
    tokenizer.return_value = {"input_ids": torch.tensor([[10, 20, 30]])}
    return tokenizer


@pytest.fixture(autouse=True)
def clean_tts_manager():
    """Réinitialise le gestionnaire de modèles avant et après chaque test."""
    reset_tts_models()
    yield
    reset_tts_models()


# ------------------------------------------------------------------ #
#  1, 2, 3. Sélection des modèles selon la langue
# ------------------------------------------------------------------ #

@pytest.mark.parametrize(
    "lang_input,expected_model,expected_canon",
    [
        ("fr", "facebook/mms-tts-fra", "fr"),
        ("fra", "facebook/mms-tts-fra", "fr"),
        ("fon", "facebook/mms-tts-fon", "fon"),
        ("yo", "facebook/mms-tts-yor", "yor"),
        ("yor", "facebook/mms-tts-yor", "yor"),
    ],
)
def test_model_selection_by_language(tmp_path: Path, lang_input: str, expected_model: str, expected_canon: str):
    fake_model = make_fake_vits_model()
    fake_tok = make_fake_tokenizer()

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok) as mock_tok, \
         mock.patch("transformers.VitsModel.from_pretrained", return_value=fake_model) as mock_mod:

        out_wav = tmp_path / f"speech_{lang_input}.wav"
        res = synthesize("Test", language=lang_input, output_path=out_wav)

        assert res["language"] == expected_canon
        assert res["model"] == expected_model
        assert mock_tok.call_args[0][0] == expected_model
        assert mock_mod.call_args[0][0] == expected_model


# ------------------------------------------------------------------ #
#  4. Génération d'un texte court
# ------------------------------------------------------------------ #

def test_short_text_generation(tmp_path: Path):
    fake_model = make_fake_vits_model()
    fake_tok = make_fake_tokenizer()

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
         mock.patch("transformers.VitsModel.from_pretrained", return_value=fake_model):

        out_wav = tmp_path / "short.wav"
        res = synthesize("Bonjour.", language="fr", output_path=out_wav)

        assert res["duration"] == pytest.approx(0.5, abs=0.05)
        assert out_wav.is_file()
        assert fake_model.call_count == 1


# ------------------------------------------------------------------ #
#  5. Gestion des textes longs avec découpage et silences
# ------------------------------------------------------------------ #

def test_long_text_splitting_and_sentence_concatenation(tmp_path: Path):
    long_text = "Première phrase à prononcer. Deuxième phrase plus longue avec des détails ! Troisième phrase finale ?"
    chunks = split_text_into_sentences(long_text, max_chars_per_chunk=60)

    assert len(chunks) == 3
    assert chunks[0] == "Première phrase à prononcer."
    assert chunks[1] == "Deuxième phrase plus longue avec des détails !"
    assert chunks[2] == "Troisième phrase finale ?"

    fake_model = make_fake_vits_model()
    fake_tok = make_fake_tokenizer()

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
         mock.patch("transformers.VitsModel.from_pretrained", return_value=fake_model):

        out_wav = tmp_path / "long.wav"
        # 3 phrases -> 3 appels au modèle
        res = synthesize(long_text, language="fr", output_path=out_wav, silence_between_sentences_s=0.2)

        assert fake_model.call_count == 3
        # 3 morceaux de 0.5s + 2 silences de 0.2s = 1.9s
        assert res["duration"] == pytest.approx(1.9, abs=0.05)


# ------------------------------------------------------------------ #
#  6. Sauvegarde des fichiers WAV au format PCM 16 bits mono
# ------------------------------------------------------------------ #

def test_wav_file_format_and_properties(tmp_path: Path):
    out_wav = tmp_path / "test_format.wav"
    samples = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)
    save_wav_pcm16(out_wav, samples, sample_rate=16000)

    assert out_wav.is_file()
    with wave.open(str(out_wav), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # 16 bits
        assert w.getframerate() == 16000
        assert w.getnframes() == 16000


# ------------------------------------------------------------------ #
#  7. Génération des métadonnées JSON
# ------------------------------------------------------------------ #

def test_metadata_json_content(tmp_path: Path):
    fake_model = make_fake_vits_model()
    fake_tok = make_fake_tokenizer()

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
         mock.patch("transformers.VitsModel.from_pretrained", return_value=fake_model):

        out_wav = tmp_path / "metadata_test.wav"
        res = synthesize("Test métadonnées.", language="fr", output_path=out_wav)

        json_file = out_wav.with_suffix(".json")
        assert json_file.is_file()

        saved = json.loads(json_file.read_text(encoding="utf-8"))
        assert saved == res
        assert saved["text"] == "Test métadonnées."
        assert saved["language"] == "fr"
        assert saved["sample_rate"] == 16000
        assert saved["duration"] > 0
        assert "created_at" in saved
        assert saved["file"] == str(out_wav)


# ------------------------------------------------------------------ #
#  8. Conservation des caractères Unicode et signes diacritiques
# ------------------------------------------------------------------ #

def test_preservation_of_unicode_and_diacritics(tmp_path: Path):
    # Phrases avec caractères spécifiques du fon et du yoruba (ɛ, ɔ, accents tonaux, sub-dots ẹ, ọ, ṣ)
    fon_text = "Mi dɔ nú we ɖɔ: é nyɔ́ tawun, kúdò azɔ̌!"
    yor_text = "Ẹ kú àárọ̀ o, báwo ni gbogbo nǹkan?"

    fake_model = make_fake_vits_model()
    fake_tok = make_fake_tokenizer()

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok), \
         mock.patch("transformers.VitsModel.from_pretrained", return_value=fake_model):

        res_fon = synthesize(fon_text, language="fon", output_path=tmp_path / "fon.wav")
        res_yor = synthesize(yor_text, language="yor", output_path=tmp_path / "yor.wav")

        assert res_fon["text"] == fon_text
        assert res_yor["text"] == yor_text

        # Vérifie que les tokens passés au tokenizer contiennent bien les caractères spéciaux et diacritiques
        all_passed_texts = " ".join(call[0][0] for call in fake_tok.call_args_list)
        # Caractères fon attendus :
        assert "ɖ" in all_passed_texts and "ɔ" in all_passed_texts and "azɔ̌" in all_passed_texts
        # Caractères yoruba attendus :
        assert "Ẹ" in all_passed_texts and "àárọ̀" in all_passed_texts and "ǹ" in all_passed_texts


# ------------------------------------------------------------------ #
#  9. Langues non prises en charge et texte vide
# ------------------------------------------------------------------ #

def test_unsupported_languages_rejected():
    with pytest.raises(UnsupportedLanguageError, match="non prise en charge"):
        synthesize("Hello", language="en")

    with pytest.raises(UnsupportedLanguageError, match="non prise en charge"):
        synthesize("Hallo", language="de")

    with pytest.raises(EmptyTextError, match="vide"):
        synthesize("   ", language="fr")


# ------------------------------------------------------------------ #
#  10. Gestion des erreurs de chargement des modèles
# ------------------------------------------------------------------ #

def test_model_loading_error_handling(tmp_path: Path):
    with mock.patch("transformers.VitsModel.from_pretrained", side_effect=OSError("Poids corrompus")):
        with pytest.raises(TTSModelLoadError, match="Impossible de charger le modèle"):
            synthesize("Test", language="fr", output_path=tmp_path / "err.wav")


# ------------------------------------------------------------------ #
#  11. Réutilisation du modèle en mémoire pour une même langue
# ------------------------------------------------------------------ #

def test_model_reuse_for_same_language(tmp_path: Path):
    fake_model = make_fake_vits_model()
    fake_tok = make_fake_tokenizer()

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok) as mock_tok, \
         mock.patch("transformers.VitsModel.from_pretrained", return_value=fake_model) as mock_mod:

        # Deux synthèses successives en français
        synthesize("Phrase un.", language="fr", output_path=tmp_path / "fr1.wav")
        synthesize("Phrase deux.", language="fr", output_path=tmp_path / "fr2.wav")

        # Le modèle et le tokenizer ne doivent être chargés qu'UNE SEULE fois
        assert mock_tok.call_count == 1
        assert mock_mod.call_count == 1


# ------------------------------------------------------------------ #
#  12. Gestion des ressources et déchargement de l'ancien modèle
# ------------------------------------------------------------------ #

def test_single_model_in_memory_and_resource_unloading(tmp_path: Path):
    manager = TTSModelManager()

    fake_model_fr = make_fake_vits_model()
    fake_tok_fr = make_fake_tokenizer()

    fake_model_fon = make_fake_vits_model()
    fake_tok_fon = make_fake_tokenizer()

    with mock.patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         mock.patch("transformers.VitsModel.from_pretrained") as mock_mod:

        mock_tok.side_effect = [fake_tok_fr, fake_tok_fon]
        mock_mod.side_effect = [fake_model_fr, fake_model_fon]

        # 1. Chargement du français
        m1, t1 = manager.get_model_and_tokenizer("fr")
        assert manager.current_language == "fr"

        # 2. Changement vers le fon -> déchargement du français et instanciation du fon
        m2, t2 = manager.get_model_and_tokenizer("fon")
        assert manager.current_language == "fon"
        assert m2 is not m1

        # 3. Libération explicite
        manager.unload()
        assert manager.current_language is None
