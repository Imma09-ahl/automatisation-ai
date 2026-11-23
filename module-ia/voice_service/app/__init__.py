"""
Package principal du Voice Service.

Ce service gère l'ensemble du pipeline audio :
  1. Détection et découpage des silences (FFmpeg + Silero VAD)
  2. Transcription audio → texte (faster-whisper / MMS-1b)
  3. Synthèse vocale texte → audio (MMS-TTS / Piper)
"""
