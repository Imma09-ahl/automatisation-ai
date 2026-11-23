"""
Appel conditionnel au LLM via Instructor.

Le LLM est appelé UNIQUEMENT lorsque des champs restent manquants
après le rapprochement avec le dictionnaire.

Instructor contraint la réponse du LLM à correspondre
exactement au schéma Pydantic ComprehensionOutput.
"""

import os
import instructor
from openai import OpenAI
from app.schemas import ComprehensionOutput, FieldMatch, FieldSource


# ------------------------------------------------------------------ #
#  Client LLM (initialisé une seule fois)
# ------------------------------------------------------------------ #

_client: instructor.Instructor | None = None


def _get_client() -> instructor.Instructor:
    """Initialise et retourne le client Instructor+OpenAI."""
    global _client
    if _client is None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            raise EnvironmentError(
                "LLM_API_KEY manquante. Ajoutez-la dans votre fichier .env."
            )
        openai_client = OpenAI(api_key=api_key, base_url=base_url)
        _client = instructor.from_openai(openai_client)
    return _client


# ------------------------------------------------------------------ #
#  Construction du prompt
# ------------------------------------------------------------------ #

def build_prompt(
    transcription: str,
    missing_fields: list[str],
    resolved_fields: dict[str, FieldMatch],
) -> str:
    """
    Construit un prompt ciblé pour le LLM.

    Le prompt informe le LLM :
    - du texte transcrit original,
    - des champs déjà résolus (ne pas modifier),
    - des champs manquants à compléter.
    """
    resolved_summary = "\n".join(
        f"  - {field}: {match.value} (score={match.score})"
        for field, match in resolved_fields.items()
        if match.value is not None
    )
    missing_summary = ", ".join(missing_fields)

    return f"""Tu analyses une transcription vocale.

Transcription : "{transcription}"

Champs déjà identifiés (ne pas modifier) :
{resolved_summary or "  (aucun)"}

Champs manquants à compléter : {missing_summary}

Extrais les informations manquantes directement depuis la transcription.
Si une information est absente, laisse le champ à null.
Réponds uniquement avec les données structurées demandées."""


# ------------------------------------------------------------------ #
#  Appel LLM conditionnel
# ------------------------------------------------------------------ #

def complete_missing_fields(
    transcription: str,
    missing_fields: list[str],
    resolved_fields: dict[str, FieldMatch],
) -> dict[str, FieldMatch]:
    """
    Complète les champs manquants en appelant le LLM via Instructor.

    Paramètres
    ----------
    transcription : str
        Texte brut de la transcription.
    missing_fields : list[str]
        Champs non résolus par le dictionnaire.
    resolved_fields : dict[str, FieldMatch]
        Champs déjà résolus (passés en contexte au LLM).

    Retourne
    --------
    dict[str, FieldMatch]
        Champs complétés avec source=LLM.
    """
    if not missing_fields:
        return {}

    client = _get_client()
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    prompt = build_prompt(transcription, missing_fields, resolved_fields)

    # Appel structuré via Instructor
    response = client.chat.completions.create(
        model=model_name,
        response_model=ComprehensionOutput,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extraction des champs manquants depuis la réponse LLM
    llm_resolved = {}
    for field in missing_fields:
        entity = response.entities.get(field)
        if entity and entity.value is not None:
            llm_resolved[field] = FieldMatch(
                value=entity.value,
                score=None,
                source=FieldSource.LLM,
            )
        else:
            llm_resolved[field] = FieldMatch(source=FieldSource.MISSING)

    return llm_resolved
