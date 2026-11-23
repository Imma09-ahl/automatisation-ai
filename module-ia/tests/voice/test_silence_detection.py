"""
Tests de la détection des silences et du découpage audio.

  - Tests unitaires : Silero VAD est remplacé par des mocks (horodatages imposés).
    Ils valident la logique du module, pas la qualité réelle du modèle.
  - Tests d'intégration (marqueur ``integration``) : vrai FFmpeg + vrai modèle Silero VAD.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path
from unittest import mock

import pytest

from app import audio_ffmpeg, vad_model
from app.audio_errors import (
    AudioConversionError,
    AudioFileNotFoundError,
    EmptyAudioFileError,
    FFmpegNotFoundError,
    InvalidVADConfigError,
    NoSpeechDetectedError,
    SegmentExtractionError,
    UnsupportedAudioFormatError,
    VADDetectionError,
)
from app.silence_detection import (
    METADATA_FILENAME,
    build_segments,
    detect_speech_segments,
    main,
    split_audio_on_silences,
)
from app.vad_config import SAMPLE_RATE, VADConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_AUDIO = Path(os.getenv("VOICE_DEMO_AUDIO", REPO_ROOT / "examples" / "audio" / "demo_parole.wav"))

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg n'est pas installé")

# Deux portions parlées imposées par le mock (en échantillons, 16 kHz) : 0,5-1,5 s et 2,25-3,0 s.
TWO_SEGMENTS = [{"start": 8000, "end": 24000}, {"start": 36000, "end": 48000}]


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

def _ffmpeg_generate(destination: Path, lavfi_source: str, seconds: float, sample_rate: int = 44100) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"{lavfi_source}:sample_rate={sample_rate}" if "sine" in lavfi_source else f"{lavfi_source}:r={sample_rate}",
         "-t", str(seconds), "-ac", "2", str(destination)],
        check=True,
    )
    return destination


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    """Fichier stéréo 44,1 kHz de 4 s (sinusoïde) : vérifie aussi la conversion mono 16 kHz."""
    return _ffmpeg_generate(tmp_path / "tonalite.wav", "sine=frequency=440", 4)


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """Fichier stéréo 44,1 kHz de 3 s entièrement silencieux."""
    return _ffmpeg_generate(tmp_path / "silence.wav", "anullsrc=cl=stereo", 3)


@pytest.fixture
def mocked_vad():
    """Remplace le modèle et ``get_speech_timestamps`` ; le vrai modèle n'est jamais chargé."""
    vad_model.reset_vad_model()
    with mock.patch.object(vad_model, "load_silero_vad", return_value=object()) as loader, \
            mock.patch.object(vad_model, "get_speech_timestamps", return_value=TWO_SEGMENTS) as detector:
        yield mock.Mock(loader=loader, detector=detector)
    vad_model.reset_vad_model()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ #
#  1. Plusieurs portions parlées
# ------------------------------------------------------------------ #

@requires_ffmpeg
def test_detects_multiple_speech_portions(tone_wav, mocked_vad):
    segments = detect_speech_segments(tone_wav, VADConfig())

    assert segments == [{"start": 0.5, "end": 1.5}, {"start": 2.25, "end": 3.0}]


@requires_ffmpeg
def test_split_creates_one_wav_per_portion(tone_wav, tmp_path, mocked_vad):
    result = split_audio_on_silences(tone_wav, tmp_path / "out", VADConfig())

    assert result["segments_count"] == 2
    output_dir = Path(result["output_dir"])
    assert output_dir == tmp_path / "out" / "tonalite"
    assert sorted(p.name for p in output_dir.glob("*.wav")) == ["segment_0001.wav", "segment_0002.wav"]


# ------------------------------------------------------------------ #
#  2. Fichier entièrement silencieux
# ------------------------------------------------------------------ #

@requires_ffmpeg
def test_silent_file_raises_no_speech_and_creates_nothing(silent_wav, tmp_path, mocked_vad):
    mocked_vad.detector.return_value = []

    with pytest.raises(NoSpeechDetectedError, match="Aucune parole"):
        split_audio_on_silences(silent_wav, tmp_path / "out", VADConfig())

    assert not (tmp_path / "out").exists()
    assert detect_speech_segments(silent_wav, VADConfig()) == []


# ------------------------------------------------------------------ #
#  3. Fichier inexistant ou vide
# ------------------------------------------------------------------ #

def test_missing_file_raises_explicit_error(tmp_path):
    with pytest.raises(AudioFileNotFoundError, match="introuvable"):
        split_audio_on_silences(tmp_path / "absent.wav", tmp_path / "out")


def test_directory_instead_of_file_is_rejected(tmp_path):
    with pytest.raises(AudioFileNotFoundError, match="pas un fichier"):
        detect_speech_segments(tmp_path)


def test_empty_file_raises_explicit_error(tmp_path):
    empty = tmp_path / "vide.wav"
    empty.touch()

    with pytest.raises(EmptyAudioFileError, match="vide"):
        split_audio_on_silences(empty, tmp_path / "out")


# ------------------------------------------------------------------ #
#  4. Erreurs FFmpeg
# ------------------------------------------------------------------ #

def test_ffmpeg_missing_raises_explicit_error(tone_wav_or_bytes, tmp_path):
    with mock.patch.object(audio_ffmpeg.shutil, "which", return_value=None):
        with pytest.raises(FFmpegNotFoundError, match="FFmpeg est introuvable"):
            split_audio_on_silences(tone_wav_or_bytes, tmp_path / "out")


@requires_ffmpeg
def test_unsupported_format_raises_explicit_error(tmp_path, mocked_vad):
    fake = tmp_path / "pas_un_audio.mp3"
    fake.write_text("ceci n'est pas un fichier audio", encoding="utf-8")

    with pytest.raises(UnsupportedAudioFormatError, match="non pris en charge"):
        split_audio_on_silences(fake, tmp_path / "out")
    mocked_vad.loader.assert_not_called()


def test_conversion_failure_raises_explicit_error(tone_wav_or_bytes, tmp_path):
    failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Erreur interne simulée")
    with mock.patch.object(audio_ffmpeg.shutil, "which", return_value="ffmpeg"), \
            mock.patch.object(audio_ffmpeg.subprocess, "run", return_value=failed):
        with pytest.raises(AudioConversionError, match="Erreur interne simulée"):
            detect_speech_segments(tone_wav_or_bytes, VADConfig())


@requires_ffmpeg
def test_segment_extraction_failure_raises_explicit_error(tone_wav, tmp_path, mocked_vad):
    real_run = audio_ffmpeg.run_ffmpeg

    def fail_on_atrim(arguments):
        if "-af" in arguments:
            return subprocess.CompletedProcess(args=arguments, returncode=1, stdout="", stderr="découpage impossible")
        return real_run(arguments)

    with mock.patch.object(audio_ffmpeg, "run_ffmpeg", side_effect=fail_on_atrim):
        with pytest.raises(SegmentExtractionError, match="segment_0001.wav"):
            split_audio_on_silences(tone_wav, tmp_path / "out", VADConfig())
    assert not (tmp_path / "out" / "tonalite" / "segment_0001.wav").exists()


@requires_ffmpeg
def test_vad_failure_raises_explicit_error(tone_wav, mocked_vad):
    mocked_vad.detector.side_effect = RuntimeError("tenseur invalide")

    with pytest.raises(VADDetectionError, match="tenseur invalide"):
        detect_speech_segments(tone_wav, VADConfig())


def test_model_loading_failure_raises_explicit_error():
    vad_model.reset_vad_model()
    with mock.patch.object(vad_model, "load_silero_vad", side_effect=OSError("poids absents")):
        with pytest.raises(VADDetectionError, match="poids absents"):
            vad_model.get_vad_model()
    vad_model.reset_vad_model()


@pytest.fixture
def tone_wav_or_bytes(tmp_path: Path) -> Path:
    """Fichier non vide utilisable même sans FFmpeg (le contenu n'est pas décodé dans ces tests)."""
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 100)
    return path


# ------------------------------------------------------------------ #
#  5. Validité des horodatages
# ------------------------------------------------------------------ #

@requires_ffmpeg
def test_timestamps_are_ordered_and_within_duration(tone_wav, tmp_path, mocked_vad):
    mocked_vad.detector.return_value = [
        {"start": 0, "end": 16000},
        {"start": 40000, "end": 10_000_000},  # dépasse la fin réelle (4 s) : doit être borné
    ]

    result = split_audio_on_silences(tone_wav, tmp_path / "out", VADConfig())

    assert result["duration"] == pytest.approx(4.0, abs=0.01)
    previous_end = 0.0
    for segment in result["segments"]:
        assert 0 <= previous_end <= segment["start"] < segment["end"] <= result["duration"]
        assert segment["duration"] == pytest.approx(segment["end"] - segment["start"], abs=0.002)
        previous_end = segment["end"]
    assert result["segments"][-1]["end"] == result["duration"]


def test_build_segments_drops_empty_or_invalid_intervals():
    segments = build_segments(
        [{"start": 100, "end": 100}, {"start": 500, "end": 200}, {"start": -50, "end": 1600}, {"start": 3000, "end": 5000}],
        total_samples=4000,
    )

    assert [(s.index, s.start_sample, s.end_sample) for s in segments] == [(1, 0, 1600), (2, 3000, 4000)]


# ------------------------------------------------------------------ #
#  6. Génération des segments WAV
# ------------------------------------------------------------------ #

@requires_ffmpeg
def test_segment_wavs_are_mono_16khz_pcm16_with_exact_length(tone_wav, tmp_path, mocked_vad):
    original_hash = _sha256(tone_wav)

    result = split_audio_on_silences(tone_wav, tmp_path / "out", VADConfig())

    for expected, segment in zip(TWO_SEGMENTS, result["segments"]):
        with wave.open(str(Path(result["output_dir"]) / segment["file"]), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, SAMPLE_RATE)
            assert wav.getnframes() == expected["end"] - expected["start"]
    assert _sha256(tone_wav) == original_hash, "le fichier original ne doit pas être modifié"


@requires_ffmpeg
def test_rerun_removes_stale_segments(tone_wav, tmp_path, mocked_vad):
    split_audio_on_silences(tone_wav, tmp_path / "out", VADConfig())
    mocked_vad.detector.return_value = TWO_SEGMENTS[:1]

    result = split_audio_on_silences(tone_wav, tmp_path / "out", VADConfig())

    assert [p.name for p in Path(result["output_dir"]).glob("*.wav")] == ["segment_0001.wav"]


# ------------------------------------------------------------------ #
#  7. Fichier metadata.json
# ------------------------------------------------------------------ #

@requires_ffmpeg
def test_metadata_json_content(tone_wav, tmp_path, mocked_vad):
    result = split_audio_on_silences(tone_wav, tmp_path / "out", VADConfig())

    metadata = json.loads((Path(result["output_dir"]) / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata == {
        "source": "tonalite.wav",
        "duration": 4.0,
        "segments_count": 2,
        "sample_rate": 16000,
        "segments": [
            {"index": 1, "start": 0.5, "end": 1.5, "duration": 1.0, "file": "segment_0001.wav"},
            {"index": 2, "start": 2.25, "end": 3.0, "duration": 0.75, "file": "segment_0002.wav"},
        ],
    }


# ------------------------------------------------------------------ #
#  8. Paramètres configurables
# ------------------------------------------------------------------ #

def test_default_config_values():
    assert VADConfig().to_dict() == {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 100,
    }


def test_config_from_environment_variables():
    config = VADConfig.from_env({"VAD_THRESHOLD": "0.7", "VAD_MIN_SILENCE_DURATION_MS": "800", "VAD_SPEECH_PAD_MS": ""})

    assert (config.threshold, config.min_silence_duration_ms, config.speech_pad_ms) == (0.7, 800, 100)


@pytest.mark.parametrize("kwargs", [{"threshold": 0}, {"threshold": 1.5}, {"speech_pad_ms": -1}, {"min_speech_duration_ms": 2.5}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(InvalidVADConfigError):
        VADConfig(**kwargs)


def test_invalid_environment_value_is_rejected():
    with pytest.raises(InvalidVADConfigError, match="VAD_THRESHOLD"):
        VADConfig.from_env({"VAD_THRESHOLD": "élevé"})


def test_with_overrides_ignores_none_and_rejects_unknown():
    config = VADConfig().with_overrides(threshold=0.6, speech_pad_ms=None)

    assert (config.threshold, config.speech_pad_ms) == (0.6, 100)
    with pytest.raises(InvalidVADConfigError, match="inconnu"):
        VADConfig().with_overrides(window=3)


@requires_ffmpeg
def test_config_is_passed_to_silero_and_model_loaded_once(tone_wav, tmp_path, mocked_vad):
    config = VADConfig(threshold=0.65, min_speech_duration_ms=300, min_silence_duration_ms=700, speech_pad_ms=40)

    detect_speech_segments(tone_wav, config)
    split_audio_on_silences(tone_wav, tmp_path / "out", config)

    mocked_vad.loader.assert_called_once()
    kwargs = mocked_vad.detector.call_args.kwargs
    assert kwargs == {
        "threshold": 0.65,
        "sampling_rate": 16000,
        "min_speech_duration_ms": 300,
        "min_silence_duration_ms": 700,
        "speech_pad_ms": 40,
        "return_seconds": False,
    }


@requires_ffmpeg
def test_cli_applies_overrides_and_reports_errors(tone_wav, tmp_path, mocked_vad, capsys):
    assert main([str(tone_wav), "--output-dir", str(tmp_path / "out"), "--threshold", "0.8"]) == 0
    assert mocked_vad.detector.call_args.kwargs["threshold"] == 0.8

    assert main([str(tmp_path / "absent.wav")]) == 1
    assert "AudioFileNotFoundError" in capsys.readouterr().err


# ------------------------------------------------------------------ #
#  Intégration : vrai FFmpeg + vrai modèle Silero VAD
# ------------------------------------------------------------------ #

@pytest.fixture
def real_vad():
    vad_model.reset_vad_model()
    yield
    vad_model.reset_vad_model()


@pytest.mark.integration
@requires_ffmpeg
def test_integration_real_model_finds_no_speech_in_silence(silent_wav, tmp_path, real_vad):
    with pytest.raises(NoSpeechDetectedError):
        split_audio_on_silences(silent_wav, tmp_path / "out", VADConfig())


@pytest.mark.integration
@requires_ffmpeg
@pytest.mark.skipif(not DEMO_AUDIO.is_file(), reason=f"fichier de démonstration absent : {DEMO_AUDIO}")
def test_integration_real_model_splits_demo_file(tmp_path, real_vad):
    original_hash = _sha256(DEMO_AUDIO)

    result = split_audio_on_silences(DEMO_AUDIO, tmp_path / "out", VADConfig())

    output_dir = Path(result["output_dir"])
    assert result["segments_count"] >= 2, "le fichier de démonstration contient deux phrases séparées par un silence"
    for segment in result["segments"]:
        path = output_dir / segment["file"]
        with wave.open(str(path), "rb") as wav:
            assert wav.getframerate() == SAMPLE_RATE and wav.getnchannels() == 1
            assert wav.getnframes() / SAMPLE_RATE == pytest.approx(segment["duration"], abs=0.002)
    metadata = json.loads((output_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["segments_count"] == result["segments_count"]
    assert _sha256(DEMO_AUDIO) == original_hash
