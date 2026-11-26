"""
Synthèse vocale multilingue (Français, Fon, Yoruba) basée sur Meta MMS-TTS.

Ce module assure :
  1. La sélection automatique du modèle MMS-TTS selon la langue demandée :
       - Français : facebook/mms-tts-fra (codes 'fr' ou 'fra')
       - Fon : facebook/mms-tts-fon (code 'fon')
       - Yoruba : facebook/mms-tts-yor (codes 'yo' ou 'yor')
  2. La préservation stricte des caractères Unicode et des signes diacritiques (ɛ, ɔ, tons, etc.).
  3. La gestion des textes longs par découpage en phrases sans coupure de mots,
     avec insertion d'un silence configurable entre les phrases.
  4. La gestion mémoire économe (un seul modèle chargé à la fois en RAM, libération automatique).
  5. La sauvegarde au format WAV PCM 16 bits et la production d'un fichier de métadonnées JSON.
  6. Le support reproductible via des graines aléatoires (seed).
  7. L'architecture extensible pour Piper en tant que moteur alternatif optionnel pour le français.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import re
import sys
import threading
import wave
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

_voice_service_dir = Path(__file__).resolve().parent.parent
if str(_voice_service_dir) not in sys.path:
    sys.path.insert(0, str(_voice_service_dir))

from app.audio_errors import (
    EmptyTextError,
    TTSGenerationError,
    TTSModelLoadError,
    TTSSaveError,
    UnsupportedLanguageError,
    VoiceProcessingError,
)

PathLike = Union[str, Path]

DEFAULT_OUTPUT_DIR = Path("outputs") / "tts"

# Modèles MMS-TTS officiels
MMS_TTS_MODELS = {
    "fr": "facebook/mms-tts-fra",
    "fra": "facebook/mms-tts-fra",
    "fon": "facebook/mms-tts-fon",
    "yo": "facebook/mms-tts-yor",
    "yor": "facebook/mms-tts-yor",
}

CANONICAL_LANG_MAP = {
    "fr": "fr",
    "fra": "fr",
    "fon": "fon",
    "yo": "yor",
    "yor": "yor",
}

DEFAULT_FILENAMES = {
    "fr": "speech_fr.wav",
    "fon": "speech_fon.wav",
    "yor": "speech_yor.wav",
}


def _resolve_default_cache_dir() -> Optional[str]:
    """Résout le dossier de cache MMS en privilégiant models/huggingface sur D: s'il existe."""
    repo_models = Path(__file__).resolve().parents[2] / "models" / "huggingface"
    if repo_models.is_dir():
        return str(repo_models)
    return os.getenv("HF_HOME")


def normalize_tts_language(language: str) -> str:
    """
    Valide et normalise le code de langue pour le TTS :
      - 'fr' ou 'fra' -> 'fr'
      - 'fon' -> 'fon'
      - 'yo' ou 'yor' -> 'yor'

    Lève ``UnsupportedLanguageError`` si la langue n'est pas prise en charge.
    """
    if not isinstance(language, str):
        raise UnsupportedLanguageError(f"Le code langue doit être une chaîne (reçu: {language!r}).")
    cleaned = language.strip().lower()
    if cleaned not in CANONICAL_LANG_MAP:
        supported = sorted(set(CANONICAL_LANG_MAP.keys()))
        raise UnsupportedLanguageError(
            f"Langue TTS non prise en charge : '{language}'. Langues disponibles : {supported}"
        )
    return CANONICAL_LANG_MAP[cleaned]


# ------------------------------------------------------------------ #
#  Gestionnaire de modèles TTS (Single-model en mémoire)
# ------------------------------------------------------------------ #

class TTSModelManager:
    """
    Gestionnaire singleton du modèle TTS en mémoire.
    Ne conserve qu'un seul modèle en mémoire vive pour éviter de saturer
    les ressources (RAM de 8 Go) et libère la mémoire lors du changement de langue.
    """

    def __init__(self) -> None:
        self._current_lang: Optional[str] = None
        self._model: Optional[VitsModel] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._lock = threading.Lock()

    def get_model_and_tokenizer(
        self,
        language: str,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> tuple[VitsModel, AutoTokenizer]:
        canonical_lang = normalize_tts_language(language)
        model_id = MMS_TTS_MODELS[canonical_lang]

        with self._lock:
            # Si le modèle chargé est déjà celui de la langue demandée, on le réutilise
            if self._current_lang == canonical_lang and self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer

            # Sinon, libérer les ressources du modèle précédent
            self._unload_locked()

            try:
                target_cache = cache_dir or _resolve_default_cache_dir()
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=target_cache,
                    local_files_only=local_files_only,
                )
                model = VitsModel.from_pretrained(
                    model_id,
                    cache_dir=target_cache,
                    local_files_only=local_files_only,
                )
                model.eval()

                self._model = model
                self._tokenizer = tokenizer
                self._current_lang = canonical_lang
                return self._model, self._tokenizer
            except Exception as exc:
                self._unload_locked()
                raise TTSModelLoadError(
                    f"Impossible de charger le modèle MMS-TTS pour '{canonical_lang}' ({model_id}) : {exc}"
                ) from exc

    def unload(self) -> None:
        """Libère explicitement le modèle chargé de la mémoire."""
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        self._model = None
        self._tokenizer = None
        self._current_lang = None
        gc.collect()

    @property
    def current_language(self) -> Optional[str]:
        return self._current_lang


_model_manager = TTSModelManager()


def get_tts_model_manager() -> TTSModelManager:
    return _model_manager


def reset_tts_models() -> None:
    """Réinitialise le gestionnaire de modèles (utile pour les tests)."""
    _model_manager.unload()


# ------------------------------------------------------------------ #
#  Découpage des textes longs
# ------------------------------------------------------------------ #

_SENTENCE_SPLIT_REGEX = re.compile(r"([.!?;:\n]+)")


def split_text_into_sentences(text: str, max_chars_per_chunk: int = 150) -> list[str]:
    """
    Découpe un texte en phrases sans couper de mots au milieu.
    Préserve l'intégralité des caractères Unicode et marques diacritiques.
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()

    # Découper selon la ponctuation forte de fin de phrase : . ! ? ; : \n
    tokens = _SENTENCE_SPLIT_REGEX.split(stripped)
    raw_sentences: list[str] = []
    current_sentence = ""

    for tok in tokens:
        if not tok:
            continue
        if _SENTENCE_SPLIT_REGEX.match(tok):
            current_sentence += tok
            if current_sentence.strip():
                raw_sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += tok

    if current_sentence.strip():
        raw_sentences.append(current_sentence.strip())

    if not raw_sentences:
        return [stripped]

    # Sous-découpage propre si une phrase individuelle dépasse max_chars_per_chunk
    chunks: list[str] = []
    for s in raw_sentences:
        if len(s) <= max_chars_per_chunk:
            chunks.append(s)
        else:
            # Découpage au niveau des espaces sans couper de mot
            words = s.split(" ")
            buf = ""
            for w in words:
                if not w:
                    continue
                if len(buf) + len(w) + 1 <= max_chars_per_chunk:
                    buf = f"{buf} {w}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = w
            if buf:
                chunks.append(buf)

    return [c for c in chunks if c.strip()]


# ------------------------------------------------------------------ #
#  Sauvegarde audio & métadonnées
# ------------------------------------------------------------------ #

def save_wav_pcm16(output_path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """
    Sauvegarde un tableau numpy float32 normalisé [-1.0, 1.0] en fichier WAV PCM 16 bits mono.
    Utilise le module standard Python ``wave`` sans nécessiter de bibliothèque externe supplémentaire.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Normalisation et conversion vers entiers signés 16 bits
        clipped = np.clip(waveform, -1.0, 1.0)
        int16_samples = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)      # 16 bits = 2 octets
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int16_samples.tobytes())

        if not output_path.is_file() or output_path.stat().st_size <= 44:
            raise TTSSaveError(f"Le fichier WAV généré est vide : '{output_path}'.")
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise TTSSaveError(f"Échec de la sauvegarde WAV dans '{output_path}' : {exc}") from exc


def build_and_save_tts_metadata(
    metadata_path: Path,
    text: str,
    language: str,
    model_name: str,
    sample_rate: int,
    duration: float,
    wav_path: Path,
) -> dict[str, Any]:
    """Construit et écrit les métadonnées de synthèse vocale au format JSON."""
    data = {
        "text": text,
        "language": language,
        "model": model_name,
        "sample_rate": sample_rate,
        "duration": round(duration, 3),
        "file": str(wav_path),
        "created_at": datetime.datetime.now().isoformat(),
    }
    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise TTSSaveError(f"Échec de l'écriture des métadonnées JSON '{metadata_path}' : {exc}") from exc
    return data


# ------------------------------------------------------------------ #
#  Synthèse vocale MMS-TTS
# ------------------------------------------------------------------ #

def _synthesize_single_chunk(
    chunk: str,
    model: VitsModel,
    tokenizer: AutoTokenizer,
) -> np.ndarray:
    """Génère la forme d'onde audio (tableau 1D float32) pour une phrase donnée."""
    inputs = tokenizer(chunk, return_tensors="pt")
    if "input_ids" in inputs and inputs["input_ids"].shape[-1] == 0:
        raise TTSGenerationError(f"Les caractères du texte ne sont pas reconnus par le tokenizer : '{chunk}'")

    with torch.no_grad():
        output = model(**inputs)

    waveform = getattr(output, "waveform", None)
    if waveform is None:
        raise TTSGenerationError("Le modèle VITS n'a retourné aucune onde audio.")

    # Convertir en tableau 1D numpy float32
    arr = waveform.squeeze().cpu().numpy().astype(np.float32)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr


def synthesize_mms_tts(
    text: str,
    language: str = "fr",
    output_path: Optional[PathLike] = None,
    seed: Optional[int] = None,
    silence_between_sentences_s: float = 0.25,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """
    Synthétise du texte en audio WAV avec Meta MMS-TTS pour le français, le fon ou le yoruba.
    """
    if not isinstance(text, str) or not text.strip():
        raise EmptyTextError("Le texte fourni pour la synthèse vocale est vide.")

    canonical_lang = normalize_tts_language(language)
    model_id = MMS_TTS_MODELS[canonical_lang]

    # Découpage des textes longs
    sentences = split_text_into_sentences(text)
    if not sentences:
        raise EmptyTextError("Aucune phrase valide à prononcer après analyse du texte.")

    # Fixer la graine aléatoire si demandée (reproductibilité)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model, tokenizer = _model_manager.get_model_and_tokenizer(
        canonical_lang,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    sample_rate = int(getattr(model.config, "sampling_rate", 16000))

    try:
        waveforms: list[np.ndarray] = []
        silence_samples = int(silence_between_sentences_s * sample_rate)
        silence = np.zeros(silence_samples, dtype=np.float32) if silence_samples > 0 else np.array([], dtype=np.float32)

        for i, sentence in enumerate(sentences):
            w = _synthesize_single_chunk(sentence, model, tokenizer)
            if len(w) > 0:
                if i > 0 and len(silence) > 0:
                    waveforms.append(silence)
                waveforms.append(w)

        if not waveforms:
            raise TTSGenerationError("La synthèse n'a produit aucun échantillon sonore.")

        full_waveform = np.concatenate(waveforms)
        duration = round(len(full_waveform) / sample_rate, 3)

        # Déterminer les chemins de sauvegarde
        if output_path is not None:
            dest_wav = Path(output_path)
        else:
            filename = DEFAULT_FILENAMES.get(canonical_lang, f"speech_{canonical_lang}.wav")
            dest_wav = DEFAULT_OUTPUT_DIR / filename

        dest_json = dest_wav.with_suffix(".json")

        save_wav_pcm16(dest_wav, full_waveform, sample_rate)
        metadata = build_and_save_tts_metadata(
            metadata_path=dest_json,
            text=text,
            language=canonical_lang,
            model_name=model_id,
            sample_rate=sample_rate,
            duration=duration,
            wav_path=dest_wav,
        )
        return metadata

    except VoiceProcessingError:
        raise
    except Exception as exc:
        raise TTSGenerationError(f"Erreur inattendue pendant la synthèse vocale MMS-TTS : {exc}") from exc


# ------------------------------------------------------------------ #
#  Point d'entrée général
# ------------------------------------------------------------------ #

def synthesize(
    text: str,
    language: str = "fr",
    output_path: Optional[PathLike] = None,
    engine: str = "mms-tts",
    seed: Optional[int] = None,
    silence_between_sentences_s: float = 0.25,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """
    Point d'entrée principal de synthèse vocale texte → audio WAV.

    Paramètres
    ----------
    text : texte à synthétiser.
    language : code langue ('fr'/'fra', 'fon', 'yo'/'yor').
    output_path : chemin du fichier WAV de sortie (ou défaut dans outputs/tts/).
    engine : moteur TTS ('mms-tts' par défaut, ou 'piper' pour le français).
    seed : graine aléatoire pour la reproductibilité.
    silence_between_sentences_s : durée du silence entre les phrases découpées.
    local_files_only : n'autorise que les modèles déjà sur disque.
    """
    canonical_lang = normalize_tts_language(language)

    if engine == "mms-tts":
        return synthesize_mms_tts(
            text=text,
            language=canonical_lang,
            output_path=output_path,
            seed=seed,
            silence_between_sentences_s=silence_between_sentences_s,
            local_files_only=local_files_only,
        )
    elif engine == "piper":
        if canonical_lang != "fr":
            raise UnsupportedLanguageError(
                f"Piper ne prend en charge que le français. Utilisez MMS-TTS pour le {canonical_lang}."
            )
        # Vérification de la disponibilité de Piper
        import shutil
        if shutil.which("piper") is None:
            raise TTSModelLoadError(
                "L'exécutable Piper est introuvable sur le système. "
                "Installez Piper ou utilisez le moteur par défaut 'mms-tts'."
            )
        raise NotImplementedError("Piper n'est pas encore configuré en local. Utilisez mms-tts.")
    else:
        raise ValueError(f"Moteur TTS inconnu : '{engine}'. Utiliser 'mms-tts'.")


# ------------------------------------------------------------------ #
#  Ligne de commande
# ------------------------------------------------------------------ #

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthèse vocale multilingue (Français, Fon, Yoruba) avec MMS-TTS."
    )
    parser.add_argument("text", help="Texte à vocaliser.")
    parser.add_argument(
        "-l", "--language",
        default="fr",
        help="Langue : 'fr'/'fra' (français), 'fon' (fon), 'yo'/'yor' (yoruba). Défaut : fr.",
    )
    parser.add_argument("-o", "--output", default=None, help="Chemin du fichier WAV de sortie.")
    parser.add_argument("--seed", type=int, default=None, help="Graine aléatoire de reproductibilité.")
    parser.add_argument("--local-files-only", action="store_true", help="Interdit tout téléchargement.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        res = synthesize(
            text=args.text,
            language=args.language,
            output_path=args.output,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    except VoiceProcessingError as exc:
        print(f"Erreur ({type(exc).__name__}) : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
