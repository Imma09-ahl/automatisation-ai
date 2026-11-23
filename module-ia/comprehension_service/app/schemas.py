"""
Schémas Pydantic du Comprehension Service.

Définit les structures de données en entrée et en sortie.
Instructor utilise ces schémas pour contraindre les réponses du LLM.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class FieldSource(str, Enum):
    """Indique comment un champ a été résolu."""
    DICTIONARY = "dictionary"   # Trouvé dans le dictionnaire (RapidFuzz)
    LLM = "llm"                 # Complété par le LLM
    MISSING = "missing"         # Introuvable


class FieldMatch(BaseModel):
    """Résultat du rapprochement pour un champ donné."""
    value: Optional[str] = Field(None, description="Valeur résolue")
    score: Optional[float] = Field(None, description="Score de similarité RapidFuzz (0-100)")
    source: FieldSource = Field(FieldSource.MISSING, description="Source de résolution")


class TranscriptionInput(BaseModel):
    """Données reçues depuis le Voice Service."""
    text: str = Field(..., description="Texte brut transcrit")
    language: str = Field("fr", description="Langue de la transcription : fr ou fon")
    segments: list[dict] = Field(default_factory=list, description="Segments horodatés")


class ComprehensionOutput(BaseModel):
    """
    Réponse finale structurée et validée.

    NOTE : Les champs métier ci-dessous sont des exemples génériques.
    Ils doivent être adaptés au schéma réel de la plateforme d'équipe.
    """
    # --- Champs métier (à adapter selon la plateforme) ---
    intent: Optional[str] = Field(None, description="Intention détectée dans la transcription")
    entities: dict[str, FieldMatch] = Field(
        default_factory=dict,
        description="Entités extraites avec leur source de résolution"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Liste des champs non résolus"
    )
    llm_used: bool = Field(False, description="Indique si le LLM a été appelé")
    raw_text: str = Field("", description="Texte brut original")
