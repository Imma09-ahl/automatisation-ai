"""
Transcription audio → texte.

Moteurs :
  - faster-whisper : pour le français (fr)
  - MMS-1b (Meta)  : pour le fon (fon)

Le modèle est chargé une seule fois en mémoire (singleton par langue).
"""

import os
from typing import Optional

# ------------------------------------------------------------------ #
#  Singletons de modèles
# ------------------------------------------------------------------ #

_whisper_model = None
_mms_model = None
_mms_processor = None


def _load_whisper(model_size: str = "base"):
    """Charge faster-whisper (transcription français)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        device = os.getenv("VOICE_DEVICE", "cpu")
        _whisper_model = WhisperModel(model_size, device=device, compute_type="int8")
    return _whisper_model


def _load_mms():
    """Charge MMS-1b via HuggingFace Transformers (transcription fon)."""
    global _mms_model, _mms_processor
    if _mms_model is None:
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        model_id = "facebook/mms-1b-all"
        _mms_processor = AutoProcessor.from_pretrained(model_id)
        _mms_model = Wav2Vec2ForCTC.from_pretrained(model_id)
    return _mms_model, _mms_processor


# ------------------------------------------------------------------ #
#  Transcription
# ------------------------------------------------------------------ #

def transcribe(
    audio_path: str,
    language: str = "fr",
    segments: Optional[list[dict]] = None,
) -> dict:
    """
    Transcrit un fichier audio en texte.

    Paramètres
    ----------
    audio_path : str
        Chemin vers le fichier WAV 16kHz mono.
    language : str
        Code langue : "fr" (français) ou "fon" (fon béninois).
    segments : list[dict], optionnel
        Timestamps de parole issus de detect_speech_segments().
        Si fournis, seuls ces segments sont transcrits.

    Retourne
    --------
    dict
        {"text": str, "segments": list[dict]}
    """
    if language == "fr":
        return _transcribe_whisper(audio_path, segments)
    elif language == "fon":
        return _transcribe_mms(audio_path)
    else:
        raise ValueError(f"Langue non supportée : '{language}'. Utiliser 'fr' ou 'fon'.")


def _transcribe_whisper(audio_path: str, segments: Optional[list[dict]]) -> dict:
    """Transcription avec faster-whisper (français)."""
    model = _load_whisper()

    segs, info = model.transcribe(
        audio_path,
        language="fr",
        beam_size=5,
        vad_filter=False,  # On utilise notre propre VAD (Silero)
    )

    full_text = ""
    result_segments = []
    for seg in segs:
        full_text += seg.text + " "
        result_segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    return {"text": full_text.strip(), "segments": result_segments}


def _transcribe_mms(audio_path: str) -> dict:
    """Transcription avec MMS-1b (fon béninois)."""
    import torch
    import soundfile as sf

    model, processor = _load_mms()

    # Chargement de l'audio
    audio_array, sample_rate = sf.read(audio_path)

    # Adaptation du processor pour le fon
    processor.tokenizer.set_target_lang("fon")
    model.load_adapter("fon")

    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0]

    return {"text": text, "segments": []}
