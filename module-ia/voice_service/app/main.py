"""
Point d'entrée du Voice Service — API FastAPI.

Routes exposées :
  GET  /health                                 → vérification de l'état du service
  POST /vad/split                              → découpe audio et retourne les segments avec URLs de téléchargement
  GET  /audio/segments/{session_id}/{filename}  → téléchargement sécurisé d'un segment audio
  POST /transcribe                             → reçoit un fichier audio, retourne le texte transcrit
  POST /synthesize                             → reçoit du texte, retourne un fichier audio WAV synthétisé
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Fallback pour exécution hors environnement FastAPI
    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

from pydantic import BaseModel, Field

import sys

_voice_service_dir = Path(__file__).resolve().parent.parent
if str(_voice_service_dir) not in sys.path:
    sys.path.insert(0, str(_voice_service_dir))

from app.audio_errors import (
    AudioFileNotFoundError,
    EmptyAudioFileError,
    NoSpeechDetectedError,
    UnsupportedLanguageError,
    VoiceProcessingError,
)
from app.silence_detection import (
    SpeechSegment,
    detect_speech_segments,
    split_audio_on_silences,
)
from app.speech_synthesis import MMS_TTS_MODELS, synthesize
from app.transcription import LANGUAGE_MAP, normalize_language_code, transcribe
from app.vad_config import VADConfig

# Configuration stockage & sécurité
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "50"))
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm", ".wma"}

SESSIONS_STORAGE_DIR = Path("outputs") / "audio" / "sessions"
SESSIONS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# --- Schémas Pydantic ---

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "voice"
    supported_languages: list[str] = ["fr", "fon", "yor"]
    version: str = "0.2.0"


class SegmentItem(BaseModel):
    index: int
    start: float
    end: float
    duration: float
    filename: str
    download_url: str


class VadSplitResponse(BaseModel):
    session_id: str
    source_file: str
    total_duration: float
    speech_segments_count: int
    segments: list[SegmentItem]


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: Optional[float] = None
    segments: list[dict[str, Any]] = Field(default_factory=list)


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "fr"
    engine: str = "mms-tts"


# --- Fonctions Utilitaires ---

def _validate_uploaded_file(file: Any) -> str:
    """Vérifie l'extension et retourne l'extension nettoyée."""
    filename = getattr(file, "filename", None) or "audio.wav"
    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format audio non supporté: '{suffix}'. Formats autorisés : {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )
    return suffix


async def _save_upload_to_temp(file: Any, suffix: str) -> str:
    """Sauvegarde temporaire avec contrôle strict de la taille maximale."""
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Le fichier audio téléversé est vide.")
    if len(content) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier audio trop volumineux ({len(content)} octets). Limite fixée à {MAX_AUDIO_SIZE_MB} Mo.",
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name


# --- Handlers d'Endpoints ---

async def health() -> HealthResponse:
    """Vérifie que le service voix est opérationnel."""
    return HealthResponse()


async def vad_split_audio(
    file: Any,
    threshold: Optional[float] = None,
) -> VadSplitResponse:
    """
    Détecte les silences et découpe un fichier audio en portions de parole.
    Stocke les segments de manière isolée et retourne des URLs de téléchargement sécurisées.
    """
    suffix = _validate_uploaded_file(file)
    tmp_path = await _save_upload_to_temp(file, suffix)

    session_id = uuid.uuid4().hex[:12]
    session_output_dir = SESSIONS_STORAGE_DIR / session_id
    session_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        vad_config = VADConfig.from_env()
        if threshold is not None:
            vad_config = vad_config.with_overrides(threshold=threshold)

        result = split_audio_on_silences(
            audio_path=tmp_path,
            output_root=SESSIONS_STORAGE_DIR,
            config=vad_config,
        )

        src_output_dir = Path(result["output_dir"])
        if src_output_dir != session_output_dir and src_output_dir.exists():
            for item in src_output_dir.iterdir():
                shutil.move(str(item), str(session_output_dir / item.name))
            shutil.rmtree(src_output_dir, ignore_errors=True)

        segments_items: list[SegmentItem] = []
        for seg in result.get("segments", []):
            fname = seg.get("file", f"segment_{seg['index']:04d}.wav")
            segments_items.append(
                SegmentItem(
                    index=seg["index"],
                    start=seg["start"],
                    end=seg["end"],
                    duration=seg["duration"],
                    filename=fname,
                    download_url=f"/audio/segments/{session_id}/{fname}",
                )
            )

        return VadSplitResponse(
            session_id=session_id,
            source_file=getattr(file, "filename", None) or "audio.wav",
            total_duration=result.get("duration", 0.0),
            speech_segments_count=len(segments_items),
            segments=segments_items,
        )
    except NoSpeechDetectedError as e:
        shutil.rmtree(session_output_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(e))
    except VoiceProcessingError as e:
        shutil.rmtree(session_output_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        shutil.rmtree(session_output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne de traitement VAD : {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def get_audio_segment(session_id: str, filename: str):
    """
    Téléchargement sécurisé d'un segment audio produit par le découpage VAD.
    Protégé contre toute tentative d'injection ou de traversée de chemin (path traversal).
    """
    safe_session = Path(session_id).name
    safe_filename = Path(filename).name

    target_file = (SESSIONS_STORAGE_DIR / safe_session / safe_filename).resolve()
    base_dir = SESSIONS_STORAGE_DIR.resolve()

    try:
        target_file.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Accès interdit au chemin demandé.")

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Segment '{safe_filename}' introuvable.")

    if HAS_FASTAPI:
        return FileResponse(path=str(target_file), media_type="audio/wav", filename=safe_filename)
    return str(target_file)


async def transcribe_audio(
    file: Any,
    language: str = "fr",
) -> TranscribeResponse:
    """
    Reçoit un fichier audio, applique la détection de silences,
    transcrit dans la langue demandée (fr, fon, yor) et retourne le texte complet.
    """
    suffix = _validate_uploaded_file(file)
    tmp_path = await _save_upload_to_temp(file, suffix)

    try:
        canon_lang = normalize_language_code(language)
    except UnsupportedLanguageError as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=str(e))

    try:
        segments = detect_speech_segments(tmp_path)
        result = transcribe(tmp_path, language=canon_lang, segments=segments)

        return TranscribeResponse(
            text=result["text"],
            language=canon_lang,
            duration=result.get("duration"),
            segments=result.get("segments", []),
        )
    except NoSpeechDetectedError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except VoiceProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne de transcription : {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def synthesize_audio(request: SynthesizeRequest):
    """
    Synthétise un texte en fichier audio WAV dans l'une des 3 langues prises en charge :
    - fr (MMS-TTS Français)
    - fon (MMS-TTS Fon)
    - yor (MMS-TTS Yoruba)
    """
    try:
        canon_lang = normalize_language_code(request.language)
    except UnsupportedLanguageError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        res = synthesize(
            text=request.text,
            language=canon_lang,
            engine=request.engine,
        )
        wav_path = res["file"] if isinstance(res, dict) else res
        if HAS_FASTAPI:
            return FileResponse(
                path=str(wav_path),
                media_type="audio/wav",
                filename=f"speech_{canon_lang}.wav",
            )
        return str(wav_path)
    except VoiceProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne de synthèse vocale : {e}")


# --- Enregistrement de l'Application FastAPI ---

if HAS_FASTAPI:
    app = FastAPI(
        title="Voice Service",
        description="Service voix : détection de silences (VAD), transcription multilingue et synthèse vocale (FR, Fon, Yoruba)",
        version="0.2.0",
    )

    app.get("/health", response_model=HealthResponse)(health)
    app.post("/vad/split", response_model=VadSplitResponse)(vad_split_audio)
    app.get("/audio/segments/{session_id}/{filename}")(get_audio_segment)
    app.post("/transcribe", response_model=TranscribeResponse)(transcribe_audio)
    app.post("/synthesize")(synthesize_audio)
else:
    app = None
