"""
Moteur de transcription pour le Fon et le Yoruba basé sur Meta MMS (facebook/mms-1b-all).

Caractéristiques :
  - Modèle : facebook/mms-1b-all (ASR fine-tuné avec adaptateurs)
  - Fon : adaptateur 'fon', processor.tokenizer.set_target_lang("fon"), model.load_adapter("fon")
  - Yoruba : adaptateur 'yor', processor.tokenizer.set_target_lang("yor"), model.load_adapter("yor")
  - Inférence sous torch.no_grad()
  - Audio mono 16 kHz
  - Modèle réutilisable (singleton thread-safe) et changement dynamique d'adaptateur
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from app.audio_errors import MMSAdapterError, MMSModelLoadError, TranscriptionError
from app.audio_ffmpeg import load_pcm_wav
from app.vad_config import SAMPLE_RATE

MMS_MODEL_ID = "facebook/mms-1b-all"
SUPPORTED_ADAPTERS = {"fon", "yor"}

_mms_model: Any = None
_mms_processor: Any = None
_current_adapter: Optional[str] = None
_load_lock = threading.Lock()
_inference_lock = threading.Lock()


def _resolve_default_cache_dir() -> Optional[str]:
    """Résout le dossier de cache MMS en privilégiant models/huggingface s'il existe."""
    repo_models = Path(__file__).resolve().parents[2] / "models" / "huggingface"
    if repo_models.is_dir():
        return str(repo_models)
    return os.getenv("HF_HOME")


def get_mms_components(
    model_id: str = MMS_MODEL_ID,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Charge ou retourne le tuple (model, processor) pour MMS-1b-all."""
    global _mms_model, _mms_processor
    with _load_lock:
        if _mms_model is None or _mms_processor is None:
            try:
                target_cache = cache_dir or _resolve_default_cache_dir()
                _mms_processor = AutoProcessor.from_pretrained(
                    model_id,
                    cache_dir=target_cache,
                    local_files_only=local_files_only,
                )
                _mms_model = Wav2Vec2ForCTC.from_pretrained(
                    model_id,
                    cache_dir=target_cache,
                    local_files_only=local_files_only,
                )
                _mms_model.eval()
            except Exception as exc:
                raise MMSModelLoadError(
                    f"Impossible de charger le modèle MMS '{model_id}' : {exc}"
                ) from exc
    return _mms_model, _mms_processor


def set_mms_adapter(model: Any, processor: Any, adapter_lang: str) -> None:
    """Active l'adaptateur linguistique demandé (fon ou yor) sur le processeur et le modèle."""
    global _current_adapter
    if adapter_lang not in SUPPORTED_ADAPTERS:
        raise MMSAdapterError(
            f"Adaptateur MMS non pris en charge : '{adapter_lang}'. Disponibles : {sorted(SUPPORTED_ADAPTERS)}"
        )
    if _current_adapter != adapter_lang:
        try:
            processor.tokenizer.set_target_lang(adapter_lang)
            model.load_adapter(adapter_lang)
            _current_adapter = adapter_lang
        except Exception as exc:
            raise MMSAdapterError(
                f"Échec du chargement de l'adaptateur MMS '{adapter_lang}' : {exc}"
            ) from exc


def reset_mms_engine() -> None:
    """Réinitialise l'instance du modèle et de l'adaptateur (utile pour les tests)."""
    global _mms_model, _mms_processor, _current_adapter
    with _load_lock:
        _mms_model = None
        _mms_processor = None
        _current_adapter = None


def get_current_adapter() -> Optional[str]:
    """Retourne l'adaptateur actuellement actif."""
    return _current_adapter


def transcribe_mms_segment(
    audio_path: Path | str,
    adapter_lang: str,
    model_id: str = MMS_MODEL_ID,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Transcrit un segment audio WAV mono 16 kHz avec MMS-1b-all et l'adaptateur donné.

    Retourne une liste de résultats :
    [{"start": 0.0, "end": <duree_segment>, "text": "<texte_transcrit>"}]
    """
    path = Path(audio_path)
    model, processor = get_mms_components(
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    with _inference_lock:
        set_mms_adapter(model, processor, adapter_lang)
        try:
            # Chargement de l'audio normalisé mono 16 kHz
            audio_array = load_pcm_wav(path, sample_rate=SAMPLE_RATE)
            segment_duration = round(len(audio_array) / SAMPLE_RATE, 3)

            inputs = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0].strip()

            if not transcription:
                return []

            return [{
                "start": 0.0,
                "end": segment_duration,
                "text": transcription,
            }]
        except Exception as exc:
            raise TranscriptionError(
                f"Échec de transcription MMS ({adapter_lang}) pour '{path.name}' : {exc}"
            ) from exc
