"""
Opérations audio réalisées avec FFmpeg.

  - conversion de n'importe quel format lisible par FFmpeg en WAV mono 16 kHz PCM 16 bits ;
  - lecture de ce WAV normalisé en tableau NumPy ;
  - extraction d'un segment (à l'échantillon près) vers un fichier WAV.

FFmpeg est appelé en ligne de commande via ``subprocess`` (aucun shell).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from app.audio_errors import (
    AudioConversionError,
    EmptyAudioFileError,
    FFmpegNotFoundError,
    SegmentExtractionError,
    UnsupportedAudioFormatError,
)
from app.vad_config import SAMPLE_RATE

WAV_HEADER_SIZE = 44

_UNSUPPORTED_FORMAT_MARKERS = (
    "invalid data found when processing input",
    "unknown format",
    "error opening input",
    "failed to read frame size",
    "could not find codec parameters",
    "does not contain any stream",
    "moov atom not found",
    "unsupported codec",
    "failed to find two consecutive mpeg audio frames",
)

_INSTALL_HINT = (
    "Installez FFmpeg puis ajoutez son dossier 'bin' au PATH "
    "(Windows : winget install Gyan.FFmpeg ; Linux : apt install ffmpeg). "
    "La variable FFMPEG_BINARY permet aussi d'indiquer l'exécutable."
)


def find_ffmpeg() -> str:
    """Retourne le chemin de l'exécutable FFmpeg ou lève ``FFmpegNotFoundError``."""
    binary = os.getenv("FFMPEG_BINARY", "ffmpeg")
    path = shutil.which(binary)
    if path is None:
        raise FFmpegNotFoundError(f"FFmpeg est introuvable ('{binary}'). {_INSTALL_HINT}")
    return path


def run_ffmpeg(arguments: list[str]) -> subprocess.CompletedProcess:
    """Exécute FFmpeg en mode silencieux et non interactif, sans lever d'erreur sur le code retour."""
    command = [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *arguments]
    try:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(f"FFmpeg n'a pas pu être lancé. {_INSTALL_HINT}") from exc


def convert_to_pcm_wav(source: Path, destination: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Convertit ``source`` en WAV mono, ``sample_rate`` Hz, PCM 16 bits dans ``destination``.

    Le fichier source n'est jamais modifié.
    """
    result = run_ffmpeg([
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(destination),
    ])
    if result.returncode != 0:
        message = _last_lines(result.stderr)
        if _looks_unsupported(result.stderr, source):
            raise UnsupportedAudioFormatError(f"Format audio non pris en charge par FFmpeg pour '{source.name}' : {message}")
        raise AudioConversionError(f"Échec de la conversion FFmpeg de '{source.name}' : {message}")
    if not destination.is_file():
        raise AudioConversionError(f"FFmpeg n'a produit aucun fichier pour '{source.name}'.")


def load_pcm_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Lit un WAV mono PCM 16 bits et retourne un tableau float32 normalisé dans [-1, 1].

    Lève ``EmptyAudioFileError`` si le fichier ne contient aucun échantillon.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioConversionError(f"WAV converti illisible : {exc}") from exc
    if (channels, width, rate) != (1, 2, sample_rate):
        raise AudioConversionError(
            f"WAV converti inattendu : {channels} canal(aux), {width * 8} bits, {rate} Hz "
            f"(attendu : mono, 16 bits, {sample_rate} Hz)."
        )
    samples = np.frombuffer(frames, dtype="<i2")
    if samples.size == 0:
        raise EmptyAudioFileError("Le fichier audio ne contient aucun échantillon exploitable.")
    return samples.astype(np.float32) / 32768.0


def extract_segment(
    source_wav: Path,
    destination: Path,
    start_sample: int,
    end_sample: int,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """
    Extrait les échantillons ``[start_sample, end_sample[`` de ``source_wav`` vers ``destination``.

    Le découpage utilise le filtre FFmpeg ``atrim`` à l'échantillon près.
    Un fichier vide ou invalide est supprimé et provoque ``SegmentExtractionError``.
    """
    if end_sample <= start_sample:
        raise SegmentExtractionError(f"Limites de segment invalides : début {start_sample}, fin {end_sample}.")
    result = run_ffmpeg([
        "-i", str(source_wav),
        "-af", f"atrim=start_sample={start_sample}:end_sample={end_sample}",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(destination),
    ])
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise SegmentExtractionError(
            f"Échec de l'extraction du segment '{destination.name}' : {_last_lines(result.stderr)}"
        )
    if not destination.is_file() or destination.stat().st_size <= WAV_HEADER_SIZE:
        destination.unlink(missing_ok=True)
        raise SegmentExtractionError(f"Le segment '{destination.name}' généré est vide.")


def _looks_unsupported(stderr: str, source: Path) -> bool:
    """Indique si l'échec vient de la lecture du fichier d'entrée (format illisible)."""
    lowered = (stderr or "").lower()
    if any(marker in lowered for marker in _UNSUPPORTED_FORMAT_MARKERS):
        return True
    source_text = str(source).lower()
    return any(source_text in line and "invalid argument" in line for line in lowered.splitlines())


def _last_lines(stderr: str, count: int = 3) -> str:
    lines = [line.strip() for line in (stderr or "").strip().splitlines() if line.strip()]
    return " | ".join(lines[-count:]) or "aucun détail fourni par FFmpeg"
