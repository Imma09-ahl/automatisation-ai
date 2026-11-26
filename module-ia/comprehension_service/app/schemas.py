"""
Schémas Pydantic du Comprehension Service.

Définit les structures de données en entrée et en sortie, garantissant :
  - La validation typée des transcriptions reçues.
  - La traçabilité de chaque information extraite (source : dictionnaire ou LLM).
  - L'uniformité stricte du résultat pour les trois langues (Français, Fon, Yoruba).
  - La contrainte des sorties structurées avec Instructor.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FieldSource(str, Enum):
    """Origine de la résolution d'un champ."""
    DICTIONARY = "dictionary"   # Correspondance trouvée dans le dictionnaire métier (RapidFuzz)
    LLM = "llm"                 # Information complétée par le modèle de langage (Instructor)
    MISSING = "missing"         # Information non résolue / manquante


class FieldMatch(BaseModel):
    """Détail du rapprochement d'un champ ou d'une entité."""
    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(..., description="Nom du champ ou entité (ex. intent, montant, destinataire)")
    value: Optional[str] = Field(None, description="Valeur reconnue dans la transcription")
    canonical_value: Optional[str] = Field(None, description="Valeur normalisée canonique issue du dictionnaire")
    score: Optional[float] = Field(None, description="Score de similarité RapidFuzz (0.0 à 100.0) ou None si LLM")
    source: FieldSource = Field(FieldSource.MISSING, description="Source d'extraction de l'information")

    @property
    def is_resolved(self) -> bool:
        """Indique si le champ a une valeur validée."""
        return self.value is not None and self.source in (FieldSource.DICTIONARY, FieldSource.LLM)


class TranscriptionInput(BaseModel):
    """Entrée reçue par le service de compréhension (fichier ou appel direct)."""
    text: str = Field(..., min_length=1, description="Texte transcrit issu du service voix")
    language: str = Field("fr", description="Langue de la transcription ('fr', 'fon', 'yo' ou 'yor')")
    source: Optional[str] = Field(None, description="Nom du fichier audio d'origine s'il est connu")
    segments: list[dict[str, Any]] = Field(default_factory=list, description="Segments audio horodatés optionnels")

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Le texte de transcription ne peut pas être vide.")
        return v.strip()


class LLMExtractionPayload(BaseModel):
    """
    Schéma Pydantic utilisé par Instructor pour extraire strictement
    les champs manquants depuis le LLM.
    """
    model_config = ConfigDict(extra="ignore")

    intent: Optional[str] = Field(None, description="Intention principale si manquante, sinon None")
    entities: dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Dictionnaire des champs manquants extraits : nom_champ -> valeur ou None",
    )


class ComprehensionOutput(BaseModel):
    """
    Résultat final structuré, enrichi et validé du service de compréhension.
    Structure identique pour toutes les langues traitées.
    """
    model_config = ConfigDict(extra="forbid")

    source: Optional[str] = Field(None, description="Nom du fichier source original")
    language: str = Field(..., description="Langue canonique traitée ('fr', 'fon', 'yor')")
    raw_text: str = Field(..., description="Texte transcrit original")
    normalized_text: str = Field(..., description="Texte après normalisation Unicode NFC et minuscules")

    intent: Optional[FieldMatch] = Field(None, description="Intention détectée")
    entities: dict[str, FieldMatch] = Field(
        default_factory=dict,
        description="Ensemble des entités et champs analysés avec leur statut et score",
    )

    available_fields: list[str] = Field(
        default_factory=list,
        description="Liste des champs résolus (trouvés dans le dictionnaire ou via LLM)",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Liste des champs indispensables qui n'ont pas pu être résolus",
    )

    llm_used: bool = Field(False, description="Indique si le LLM a été sollicité pour compléter des champs")
    is_complete: bool = Field(False, description="Vrai si tous les champs requis sont résolus")
