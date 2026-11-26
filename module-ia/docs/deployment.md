# Guide de Déploiement — Module IA (`module-ia`)

Ce document détaille les architectures de déploiement en conteneurs Docker pour les deux microservices de `module-ia`.

---

## 1. Architecture des Conteneurs

Afin de respecter le principe d'isolation stricte des responsabilités et des ressources :

| Conteneur | Service | Port | Poids Image | Dépendances Clés | Rôle |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`module_ia_voice`** | `voice_service` | `8001` | ~2.5 Go | Python 3.11, FFmpeg, PyTorch, Silero VAD, Transformers | Traitement audio lourd, transcription ASR, synthèse vocale TTS |
| **`module_ia_comprehension`** | `comprehension_service` | `8002` | ~150 Mo | Python 3.11, RapidFuzz, Pydantic, Instructor, FastAPI | Traitement lexical ultra-léger, validation, appel LLM conditionnel |

---

## 2. Déploiement avec Docker Compose (Recommandé)

Le fichier racine `docker-compose.yml` orchestre les deux services sur un réseau de pont privé `module_ia_net`.

### Lancer l'ensemble des services
```bash
docker compose up --build -d
```

### Vérifier l'état et les logs
```bash
# Vérifier l'état de santé (Healthchecks)
docker compose ps

# Suivre les logs en temps réel
docker compose logs -f voice-service
docker compose logs -f comprehension-service
```

### Arrêter les conteneurs
```bash
docker compose down
```

---

## 3. Lancer les Services de Façon Isolée

Si la plateforme principale héberge les services sur des serveurs physiques distincts :

### Démarrer uniquement le Service Voix
```bash
docker compose up --build -d voice-service
```
Ou via commande Docker directe :
```bash
docker build -t module_ia_voice:latest -f voice_service/Dockerfile voice_service/

docker run -d \
  --name module_ia_voice \
  -p 8001:8001 \
  -v $(pwd)/models:/models \
  -v $(pwd)/outputs:/app/outputs \
  module_ia_voice:latest
```

### Démarrer uniquement le Service Compréhension
```bash
docker compose up --build -d comprehension-service
```
Ou via commande Docker directe :
```bash
docker build -t module_ia_comprehension:latest -f comprehension_service/Dockerfile comprehension_service/

docker run -d \
  --name module_ia_comprehension \
  -p 8002:8002 \
  -e LLM_API_KEY="votre_cle" \
  -v $(pwd)/comprehension_service/data/dictionaries:/app/data/dictionaries \
  module_ia_comprehension:latest
```

---

## 4. Gestion des Volumes & Persistance

1. **Volume `/models` (Service Voix) :**  
   - Les modèles `facebook/mms-1b-all` et `facebook/mms-tts-*` sont volumineux.
   - Le montage du dossier local `./models:/models` avec la variable `HF_HOME=/models/huggingface` garantit que les poids ne sont **jamais retéléchargés** lors d'un redémarrage du conteneur.
   - Ne placez **aucun téléchargement de modèle dans les instructions RUN du Dockerfile** (évite d'alourdir inutilement les images Docker).

2. **Volume `/app/data/dictionaries` (Service Compréhension) :**  
   - Permet à l'équipe métier de modifier, enrichir ou remplacer les fichiers `fr.json`, `fon.json`, `yor.json` sans reconstruire l'image du conteneur.

3. **Volume `/app/outputs` (Service Voix) :**  
   - Stockage des segments de parole découpés par le VAD et des fichiers WAV synthétisés.

---

## 5. Recommandations de Ressources & Production

* **CPU / RAM :**
  - `voice_service` : minimum 4 cœurs CPU et 6 Go de RAM alloués.
  - `comprehension_service` : 1 cœur CPU et 512 Mo de RAM suffisent largement.
* **Support GPU Nvidia (Optionnel) :**
  - Si un GPU Nvidia est présent sur le serveur de production, installez le `nvidia-container-toolkit` et passez `VOICE_DEVICE=cuda` dans le `.env` pour diviser les temps de transcription par 4.
* **Sécurité & Proxy Inverse :**
  - En production, il est recommandé de placer un reverse proxy (Nginx, Traefik) devant les ports 8001 et 8002 avec chiffrement TLS/HTTPS et injection d'un header d'authentification (`X-API-Key`).
