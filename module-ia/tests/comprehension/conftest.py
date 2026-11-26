"""Configuration pytest pour les tests du comprehension_service."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPREHENSION_SERVICE_DIR = REPO_ROOT / "comprehension_service"

# Assurer l'isolation du namespace 'app' pour comprehension_service
if "app" in sys.modules:
    app_file = getattr(sys.modules["app"], "__file__", "")
    if app_file and "comprehension_service" not in app_file:
        for k in list(sys.modules.keys()):
            if k == "app" or k.startswith("app."):
                del sys.modules[k]

if str(COMPREHENSION_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPREHENSION_SERVICE_DIR))
