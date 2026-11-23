"""
Tests de la transcription française (faster-whisper).

  - Tests unitaires : ``WhisperModel`` est remplacé par un faux modèle. Ils valident la
    logique du module (lecture/écriture des fichiers, erreurs, paramètres), pas la
    qualité réelle de la transcription.
  - Test d'intégration (marqueur ``integration``) : vrai FFmpeg + Silero VAD + faster-whisper.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from app import vad_model, whisper_model
from app.audio_errors import (
    EmptyTranscriptionError,
    InvalidWhisperConfigError,
    MetadataError,
    TranscriptionError,
    WhisperModelLoadError,
)
from app.silence_detection import METADATA_FILENAME
from app.transcription_fr import TRANSCRIPT_FILENAME, main, split_and_transcribe_fr, transcribe_segments_dir
from app.vad_config import VADConfig
from app.whisper_config import WhisperConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_AUDIO = Path(os.getenv("VOICE_DEMO_AUDIO", REPO_ROOT / "examples" / "audio" / "demo_parole.wav"))

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg n'est pas installé")

SEGMENTS_METADATA = {
    "source": "conversation.wav",
    "duration": 3.0,
    "segments_count": 2,
    "sample_rate": 16000,
    "segments": [
        {"index": 1, "start": 0.2, "end": 1.0, "duration": 0.8, "file": "segment_0001.wav"},
        {"index": 2, "start": 1.8, "end": 2.6, "duration": 0.8, "file": "segment_0002.wav"},
    ],
}


# ------------------------------------------------------------------ #
#  Faux modèle et fixtures
# ------------------------------------------------------------------ #

class FakeWhisperModel:
    """Imite ``WhisperModel.transcribe`` : texte imposé par nom de fichier."""

    def __init__(self, texts_by_file: dict[str, list[str]]):
        self.texts_by_file = texts_by_file
        self.calls: list[tuple[str, dict]] = []
        self.fail_on: dict[str, Exception] = {}

    def transcribe(self, audio, **kwargs):
        name = Path(audio).name
        self.calls.append((name, kwargs))

        def pieces():
            if name in self.fail_on:
                raise self.fail_on[name]
            for i, text in enumerate(self.texts_by_file.get(name, [])):
                yield SimpleNamespace(start=float(i), end=float(i + 1), text=text)

        return pieces(), SimpleNamespace(language="fr", language_probability=1.0)


@pytest.fixture
def fake_whisper():
    """Remplace ``WhisperModel`` ; le vrai modèle n'est jamais chargé."""
    fake = FakeWhisperModel({"segment_0001.wav": [" Bonjour à tous."], "segment_0002.wav": [" Voici", " la suite. "]})
    whisper_model.reset_whisper_model()
    with mock.patch.object(whisper_model, "WhisperModel", return_value=fake) as loader:
        fake.loader = loader
        yield fake
    whisper_model.reset_whisper_model()


def _write_silent_wav(path: Path, seconds: float = 0.8) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))


@pytest.fixture
def segments_dir(tmp_path: Path) -> Path:
    """Dossier identique à celui produit par la détection des silences."""
    directory = tmp_path / "conversation"
    directory.mkdir()
    for segment in SEGMENTS_METADATA["segments"]:
        _write_silent_wav(directory / segment["file"])
    (directory / METADATA_FILENAME).write_text(json.dumps(SEGMENTS_METADATA), encoding="utf-8")
    return directory


def _read_metadata(directory: Path) -> dict:
    return json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ #
#  Transcription d'un dossier de segments
# ------------------------------------------------------------------ #

def test_transcribes_each_segment_and_writes_results(segments_dir, fake_whisper):
    result = transcribe_segments_dir(segments_dir, WhisperConfig())

    assert [s["text"] for s in result["segments"]] == ["Bonjour à tous.", "Voici la suite."]
    assert result["transcription"] == {
        "language": "fr",
        "engine": "faster-whisper",
        "model": "small",
        "text": "Bonjour à tous. Voici la suite.",
    }
    assert result["output_dir"] == str(segments_dir)
    written = _read_metadata(segments_dir)
    assert written == {k: v for k, v in result.items() if k != "output_dir"}
    assert (segments_dir / TRANSCRIPT_FILENAME).read_text(encoding="utf-8") == "Bonjour à tous. Voici la suite.\n"


def test_silence_detection_fields_are_kept(segments_dir, fake_whisper):
    result = transcribe_segments_dir(segments_dir, WhisperConfig())

    for original, segment in zip(SEGMENTS_METADATA["segments"], result["segments"]):
        assert {k: segment[k] for k in original} == original
    for key in ("source", "duration", "segments_count", "sample_rate"):
        assert result[key] == SEGMENTS_METADATA[key]


def test_whisper_is_called_in_french_without_internal_vad(segments_dir, fake_whisper):
    transcribe_segments_dir(segments_dir, WhisperConfig(beam_size=3))

    assert [name for name, _ in fake_whisper.calls] == ["segment_0001.wav", "segment_0002.wav"]
    for _, kwargs in fake_whisper.calls:
        assert kwargs == {"language": "fr", "beam_size": 3, "vad_filter": False}


def test_segment_without_speech_keeps_empty_text(segments_dir, fake_whisper):
    fake_whisper.texts_by_file["segment_0002.wav"] = [" "]

    result = transcribe_segments_dir(segments_dir, WhisperConfig())

    assert [s["text"] for s in result["segments"]] == ["Bonjour à tous.", ""]
    assert result["transcription"]["text"] == "Bonjour à tous."


def test_no_text_at_all_raises_and_leaves_metadata_untouched(segments_dir, fake_whisper):
    fake_whisper.texts_by_file = {}

    with pytest.raises(EmptyTranscriptionError, match="Aucun texte"):
        transcribe_segments_dir(segments_dir, WhisperConfig())

    assert _read_metadata(segments_dir) == SEGMENTS_METADATA
    assert not (segments_dir / TRANSCRIPT_FILENAME).exists()


# ------------------------------------------------------------------ #
#  Erreurs
# ------------------------------------------------------------------ #

def test_missing_metadata_raises_explicit_error(tmp_path, fake_whisper):
    with pytest.raises(MetadataError, match="introuvable"):
        transcribe_segments_dir(tmp_path, WhisperConfig())


@pytest.mark.parametrize(
    "content, message",
    [("{pas du json", "illisible"), (json.dumps({"segments": []}), "aucun segment"), (json.dumps({"segments": [{"index": 1}]}), "mal formé")],
)
def test_invalid_metadata_raises_explicit_error(segments_dir, fake_whisper, content, message):
    (segments_dir / METADATA_FILENAME).write_text(content, encoding="utf-8")

    with pytest.raises(MetadataError, match=message):
        transcribe_segments_dir(segments_dir, WhisperConfig())


def test_missing_segment_file_raises_before_transcribing(segments_dir, fake_whisper):
    (segments_dir / "segment_0002.wav").unlink()

    with pytest.raises(MetadataError, match="segment_0002.wav"):
        transcribe_segments_dir(segments_dir, WhisperConfig())
    assert fake_whisper.calls == []


def test_transcription_failure_names_segment_and_leaves_metadata_untouched(segments_dir, fake_whisper):
    fake_whisper.fail_on["segment_0002.wav"] = RuntimeError("décodage impossible")

    with pytest.raises(TranscriptionError, match="segment_0002.wav.*décodage impossible"):
        transcribe_segments_dir(segments_dir, WhisperConfig())

    assert _read_metadata(segments_dir) == SEGMENTS_METADATA
    assert not (segments_dir / TRANSCRIPT_FILENAME).exists()


@pytest.mark.parametrize("local_only, hint", [(True, "désactivez local_files_only"), (False, "connexion internet")])
def test_model_loading_failure_raises_explicit_error(segments_dir, local_only, hint):
    whisper_model.reset_whisper_model()
    with mock.patch.object(whisper_model, "WhisperModel", side_effect=OSError("poids absents")):
        with pytest.raises(WhisperModelLoadError, match=f"poids absents.*{hint}"):
            transcribe_segments_dir(segments_dir, WhisperConfig(local_files_only=local_only))
    whisper_model.reset_whisper_model()


# ------------------------------------------------------------------ #
#  Chargement du modèle
# ------------------------------------------------------------------ #

def test_model_is_loaded_once_and_reloaded_when_settings_change(segments_dir, fake_whisper):
    transcribe_segments_dir(segments_dir, WhisperConfig())
    transcribe_segments_dir(segments_dir, WhisperConfig(beam_size=2))  # beam_size ne recharge pas
    assert fake_whisper.loader.call_count == 1

    transcribe_segments_dir(segments_dir, WhisperConfig(model_size="base"))
    assert fake_whisper.loader.call_count == 2
    assert fake_whisper.loader.call_args.args == ("base",)
    assert fake_whisper.loader.call_args.kwargs == {
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 0,
        "download_root": None,
        "local_files_only": False,
    }


# ------------------------------------------------------------------ #
#  Paramètres configurables
# ------------------------------------------------------------------ #

def test_default_config_values():
    assert WhisperConfig().to_dict() == {
        "model_size": "small",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 5,
        "cpu_threads": 0,
        "download_root": None,
        "local_files_only": False,
    }


def test_config_from_environment_variables():
    config = WhisperConfig.from_env(
        {"WHISPER_MODEL_SIZE": "base", "WHISPER_BEAM_SIZE": "3", "WHISPER_LOCAL_FILES_ONLY": "oui", "WHISPER_DEVICE": ""}
    )

    assert (config.model_size, config.beam_size, config.local_files_only, config.device) == ("base", 3, True, "cpu")


@pytest.mark.parametrize(
    "kwargs",
    [{"model_size": "énorme"}, {"model_size": ""}, {"device": "tpu"}, {"beam_size": 0}, {"cpu_threads": -1}, {"local_files_only": "oui"}],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(InvalidWhisperConfigError):
        WhisperConfig(**kwargs)


def test_model_directory_path_is_accepted(tmp_path):
    assert WhisperConfig(model_size=str(tmp_path)).model_size == str(tmp_path)


@pytest.mark.parametrize("variable, value", [("WHISPER_BEAM_SIZE", "cinq"), ("WHISPER_LOCAL_FILES_ONLY", "peut-être")])
def test_invalid_environment_value_is_rejected(variable, value):
    with pytest.raises(InvalidWhisperConfigError, match=variable):
        WhisperConfig.from_env({variable: value})


def test_with_overrides_ignores_none_and_rejects_unknown():
    config = WhisperConfig().with_overrides(model_size="base", beam_size=None)

    assert (config.model_size, config.beam_size) == ("base", 5)
    with pytest.raises(InvalidWhisperConfigError, match="inconnu"):
        WhisperConfig().with_overrides(temperature=0.2)


# ------------------------------------------------------------------ #
#  Ligne de commande
# ------------------------------------------------------------------ #

def test_cli_transcribes_existing_segments_dir(segments_dir, fake_whisper, capsys):
    assert main(["--segments-dir", str(segments_dir), "--beam-size", "2"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["transcription"]["text"] == "Bonjour à tous. Voici la suite."
    assert fake_whisper.calls[0][1]["beam_size"] == 2


def test_cli_reports_errors(tmp_path, fake_whisper, capsys):
    assert main(["--segments-dir", str(tmp_path)]) == 1
    assert "MetadataError" in capsys.readouterr().err


# ------------------------------------------------------------------ #
#  Pipeline complet (FFmpeg réel, VAD et Whisper simulés)
# ------------------------------------------------------------------ #

@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    path = tmp_path / "conversation.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=44100", "-t", "3", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def mocked_vad():
    vad_model.reset_vad_model()
    timestamps = [{"start": 3200, "end": 16000}, {"start": 28800, "end": 41600}]
    with mock.patch.object(vad_model, "load_silero_vad", return_value=object()), \
            mock.patch.object(vad_model, "get_speech_timestamps", return_value=timestamps):
        yield
    vad_model.reset_vad_model()


@requires_ffmpeg
def test_split_and_transcribe_pipeline(tone_wav, tmp_path, mocked_vad, fake_whisper):
    output_root = tmp_path / "out"
    stale = output_root / "conversation" / TRANSCRIPT_FILENAME
    stale.parent.mkdir(parents=True)
    stale.write_text("ancien résultat", encoding="utf-8")

    result = split_and_transcribe_fr(tone_wav, output_root, VADConfig(), WhisperConfig())

    assert result["segments_count"] == 2
    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == [
        (0.2, 1.0, "Bonjour à tous."),
        (1.8, 2.6, "Voici la suite."),
    ]
    assert stale.read_text(encoding="utf-8") == "Bonjour à tous. Voici la suite.\n"


# ------------------------------------------------------------------ #
#  Intégration : vrai FFmpeg + Silero VAD + faster-whisper
# ------------------------------------------------------------------ #

def _real_model_available() -> bool:
    whisper_model.reset_whisper_model()
    try:
        whisper_model.get_whisper_model(WhisperConfig.from_env().with_overrides(local_files_only=True))
        return True
    except WhisperModelLoadError:
        return False


@pytest.mark.integration
@requires_ffmpeg
@pytest.mark.skipif(not DEMO_AUDIO.is_file(), reason=f"fichier de démonstration absent : {DEMO_AUDIO}")
def test_integration_real_transcription_of_demo_file(tmp_path):
    if not _real_model_available():
        pytest.skip("modèle faster-whisper absent du cache local (aucun téléchargement pendant les tests)")
    vad_model.reset_vad_model()

    result = split_and_transcribe_fr(DEMO_AUDIO, tmp_path, VADConfig(), WhisperConfig.from_env().with_overrides(local_files_only=True))

    output_dir = Path(result["output_dir"])
    assert result["segments_count"] >= 1
    assert all(segment["text"] for segment in result["segments"])
    assert "bonjour" in result["transcription"]["text"].lower()
    assert _read_metadata(output_dir)["transcription"] == result["transcription"]
    assert (output_dir / TRANSCRIPT_FILENAME).read_text(encoding="utf-8").strip() == result["transcription"]["text"]
    whisper_model.reset_whisper_model()
    vad_model.reset_vad_model()
