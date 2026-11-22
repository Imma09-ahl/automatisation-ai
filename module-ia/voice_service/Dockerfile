# Dockerfile du service voix.
# Ce service est isolé dans son propre conteneur car il est le plus
# gourmand en ressources (FFmpeg, Silero VAD, faster-whisper, MMS, TTS).
#
# À définir lors de l'implémentation :
#   - image de base (CPU ou GPU/CUDA)
#   - installation de FFmpeg
#   - installation des dépendances (requirements.txt)
#   - gestion des modèles (volume monté, pas de modèles dans l'image)
#   - commande de lancement de l'application
