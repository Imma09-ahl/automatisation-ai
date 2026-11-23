"""Configuration pytest des tests du service voix."""

import sys
from pathlib import Path

VOICE_SERVICE_DIR = Path(__file__).resolve().parents[2] / "voice_service"
if str(VOICE_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_SERVICE_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: test utilisant FFmpeg et le vrai modèle Silero VAD")
