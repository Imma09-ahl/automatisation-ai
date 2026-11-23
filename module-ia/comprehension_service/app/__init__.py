"""
Package principal du Comprehension Service.

Ce service analyse les transcriptions brutes issues du Voice Service :
  1. Rapprochement avec un dictionnaire métier (RapidFuzz)
  2. Identification des champs manquants
  3. Appel conditionnel à un LLM pour compléter les données manquantes
  4. Structuration et validation de la réponse finale (Instructor + Pydantic)
"""
