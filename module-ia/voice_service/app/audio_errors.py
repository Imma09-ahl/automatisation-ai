"""
Exceptions explicites du traitement audio (détection des silences, découpage, transcription).

Toutes héritent de ``VoiceProcessingError`` : une API peut donc intercepter
une seule classe tout en conservant un message précis pour chaque cas.
"""


class VoiceProcessingError(Exception):
    """Erreur de base du traitement audio du service voix."""


class InvalidVADConfigError(VoiceProcessingError, ValueError):
    """Un paramètre de configuration du VAD est invalide."""


class AudioFileNotFoundError(VoiceProcessingError, FileNotFoundError):
    """Le fichier audio demandé n'existe pas."""


class EmptyAudioFileError(VoiceProcessingError):
    """Le fichier audio est vide ou ne contient aucun échantillon."""


class FFmpegNotFoundError(VoiceProcessingError):
    """FFmpeg n'est pas installé ou n'est pas accessible dans le PATH."""


class UnsupportedAudioFormatError(VoiceProcessingError):
    """FFmpeg ne sait pas lire le format du fichier fourni."""


class AudioConversionError(VoiceProcessingError):
    """La conversion FFmpeg vers WAV mono 16 kHz PCM 16 bits a échoué."""


class VADDetectionError(VoiceProcessingError):
    """Le chargement du modèle Silero VAD ou la détection a échoué."""


class NoSpeechDetectedError(VoiceProcessingError):
    """Aucune portion de parole n'a été détectée dans l'audio."""


class SegmentExtractionError(VoiceProcessingError):
    """L'extraction d'un segment WAV par FFmpeg a échoué."""


class InvalidWhisperConfigError(VoiceProcessingError, ValueError):
    """Un paramètre de configuration de faster-whisper est invalide."""


class WhisperModelLoadError(VoiceProcessingError):
    """Le modèle faster-whisper n'a pas pu être chargé ou téléchargé."""


class MetadataError(VoiceProcessingError):
    """``metadata.json`` ou un fichier de segment est absent ou invalide."""


class TranscriptionError(VoiceProcessingError):
    """La transcription d'un segment par faster-whisper a échoué."""


class EmptyTranscriptionError(VoiceProcessingError):
    """Aucun texte n'a été reconnu dans l'ensemble des segments."""
