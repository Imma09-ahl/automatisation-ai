"""
Point d'entrée du Voice Service — API FastAPI.

Routes exposées :
  POST /transcribe  → reçoit un fichier audio, retourne le texte transcrit
  POST /synthesize  → reçoit du texte, retourne un fichier audio
  GET  /health      → vérification de l'état du service
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os

from app.silence_detection import detect_speech_segments
from app.transcription import transcribe
from app.speech_synthesis import synthesize

app = FastAPI(
    title="Voice Service",
    description="Service voix : détection de silences, transcription, synthèse vocale",
    version="0.1.0",
)


class TranscribeResponse(BaseModel):
    text: str
    language: str
    segments: list[dict] = []


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "fr"
    engine: str = "mms-tts"


@app.get("/health")
async def health():
    """Vérifie que le service est opérationnel."""
    return {"status": "ok", "service": "voice"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "fr",
):
    """
    Reçoit un fichier audio (WAV, MP3, OGG…),
    découpe les silences, transcrit et retourne le texte.
    """
    # Sauvegarde temporaire du fichier uploadé
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Détection des segments de parole
        segments = detect_speech_segments(tmp_path)

        # Transcription
        result = transcribe(tmp_path, language=language, segments=segments)

        return TranscribeResponse(
            text=result["text"],
            language=language,
            segments=result.get("segments", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/synthesize")
async def synthesize_audio(request: SynthesizeRequest):
    """
    Reçoit du texte et retourne un fichier audio synthétisé.
    """
    try:
        output_path = synthesize(
            text=request.text,
            language=request.language,
            engine=request.engine,
        )
        return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
