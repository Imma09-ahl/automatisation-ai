"""
Point d'entrée du Comprehension Service — API FastAPI.

Routes exposées :
  POST /understand  → reçoit une transcription, retourne un objet structuré
  GET  /health      → vérification de l'état du service
"""

import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.schemas import TranscriptionInput, ComprehensionOutput, FieldSource
from app.dictionary_matching import resolve_fields
from app.llm_service import complete_missing_fields

load_dotenv()

app = FastAPI(
    title="Comprehension Service",
    description="Analyse et structuration des transcriptions vocales",
    version="0.1.0",
)

# Champs métier attendus (à adapter selon la plateforme d'équipe)
# Ces champs peuvent aussi être chargés depuis un fichier de configuration
EXPECTED_FIELDS: list[str] = os.getenv(
    "EXPECTED_FIELDS", "intent,entity_name,entity_value"
).split(",")


@app.get("/health")
async def health():
    """Vérifie que le service est opérationnel."""
    return {"status": "ok", "service": "comprehension"}


@app.post("/understand", response_model=ComprehensionOutput)
async def understand(payload: TranscriptionInput) -> ComprehensionOutput:
    """
    Analyse une transcription et retourne les données structurées.

    Pipeline :
      1. Rapprochement avec le dictionnaire (RapidFuzz).
      2. Identification des champs manquants.
      3. Appel LLM conditionnel si des champs manquent.
      4. Fusion et retour du résultat validé.
    """
    try:
        # Étape 1 : Rapprochement avec le dictionnaire
        resolved, missing = resolve_fields(
            transcription_text=payload.text,
            expected_fields=EXPECTED_FIELDS,
        )

        llm_used = False

        # Étape 2 : Appel LLM uniquement si des champs sont manquants
        if missing:
            llm_resolved = complete_missing_fields(
                transcription=payload.text,
                missing_fields=missing,
                resolved_fields=resolved,
            )
            # Mise à jour des champs avec les résultats LLM
            resolved.update(llm_resolved)
            llm_used = True
            # Recalcul des champs encore manquants
            missing = [
                f for f, m in resolved.items()
                if m.source == FieldSource.MISSING
            ]

        return ComprehensionOutput(
            intent=resolved.get("intent", None) and resolved["intent"].value,
            entities=resolved,
            missing_fields=missing,
            llm_used=llm_used,
            raw_text=payload.text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
