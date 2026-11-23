"""
Synthèse vocale texte → audio.

Moteurs disponibles :
  - "mms-tts" : MMS-TTS (Meta) via HuggingFace — couverture multilingue (fr + fon).
  - "piper"   : Piper TTS — rapide et léger, français principalement.

Le moteur est sélectionné selon le paramètre `engine`.
Par défaut : mms-tts (compatible fr et fon).
"""

import os
import tempfile


# ------------------------------------------------------------------ #
#  Singletons de modèles TTS
# ------------------------------------------------------------------ #

_mms_tts_models: dict = {}  # clé = langue


def _load_mms_tts(language: str):
    """Charge le modèle MMS-TTS pour une langue donnée."""
    if language not in _mms_tts_models:
        from transformers import VitsModel, AutoTokenizer
        # Mapping langue → identifiant modèle HuggingFace MMS-TTS
        model_map = {
            "fr": "facebook/mms-tts-fra",
            "fon": "facebook/mms-tts-fon",
        }
        model_id = model_map.get(language)
        if model_id is None:
            raise ValueError(f"Langue TTS non supportée : '{language}'")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id)
        _mms_tts_models[language] = (model, tokenizer)

    return _mms_tts_models[language]


# ------------------------------------------------------------------ #
#  Synthèse vocale
# ------------------------------------------------------------------ #

def synthesize(text: str, language: str = "fr", engine: str = "mms-tts") -> str:
    """
    Convertit du texte en fichier audio WAV.

    Paramètres
    ----------
    text : str
        Texte à synthétiser.
    language : str
        Code langue : "fr" ou "fon".
    engine : str
        Moteur TTS : "mms-tts" ou "piper".

    Retourne
    --------
    str
        Chemin vers le fichier WAV généré (temporaire).
    """
    if engine == "mms-tts":
        return _synthesize_mms_tts(text, language)
    elif engine == "piper":
        return _synthesize_piper(text, language)
    else:
        raise ValueError(f"Moteur TTS inconnu : '{engine}'. Utiliser 'mms-tts' ou 'piper'.")


def _synthesize_mms_tts(text: str, language: str) -> str:
    """Synthèse avec MMS-TTS (Meta HuggingFace)."""
    import torch
    import scipy.io.wavfile as wavfile
    import numpy as np

    model, tokenizer = _load_mms_tts(language)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform[0].numpy()
    sample_rate = model.config.sampling_rate

    # Sauvegarde dans un fichier temporaire
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, sample_rate, waveform)
    return tmp.name


def _synthesize_piper(text: str, language: str) -> str:
    """
    Synthèse avec Piper TTS (via subprocess).
    Piper doit être installé et accessible dans le PATH.
    """
    import subprocess

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    voice_map = {"fr": "fr_FR-upmc-medium"}
    voice = voice_map.get(language, "fr_FR-upmc-medium")

    cmd = ["piper", "--model", voice, "--output_file", tmp.name]
    result = subprocess.run(cmd, input=text, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Erreur Piper TTS :\n{result.stderr}")

    return tmp.name
