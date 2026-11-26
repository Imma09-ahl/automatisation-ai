# Guide de Transmission — Module IA (`module-ia`)

Ce document est destiné au **chef de groupe** et aux **développeurs de la plateforme principale** pour la prise en main, l'intégration et l'exploitation des services de `module-ia`.

---

## 1. Vue d'Ensemble & Périmètre du Module IA

Le projet `module-ia` est un dépôt autonome contenant deux microservices indépendants, conçus pour doter la plateforme collective de capacités vocales et linguistiques en trois langues : **Français (`fr`)**, **Fon (`fon`)** et **Yoruba (`yor`)**.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              MODULE IA                                 │
├───────────────────────────────────┬────────────────────────────────────┤
│   SERVICE VOIX (Port 8001)        │   SERVICE COMPRÉHENSION (Port 8002)│
├───────────────────────────────────┼────────────────────────────────────┤
│ • Détection des silences VAD      │ • Normalisation Unicode NFC        │
│ • Transcription ASR (FR, FON, YOR)│ • Rapprochement dictionnaire       │
│ • Synthèse vocale TTS (FR,FON,YOR)│ • Déclenchement LLM conditionnel   │
│ • Gestion des segments audio      │ • Validation Pydantic stricte      │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 2. Procédure Rapide de Prise en Main (Quickstart)

### Étape 1 : Récupération du dépôt
```bash
git clone https://github.com/votre-organisation/module-ia.git
cd module-ia
```

### Étape 2 : Configuration d'environnement
```bash
# Copier le template de configuration
cp .env.example .env

# Renseigner la clé LLM (si complétion des champs manquants souhaitée)
# LLM_API_KEY=sk-...
```

### Étape 3 : Démarrage avec Docker Compose (Méthode recommandée)
```bash
docker compose up --build -d
```
Les deux services sont immédiatement opérationnels :
- Service Voix sur [http://localhost:8001](http://localhost:8001)
- Service Compréhension sur [http://localhost:8002](http://localhost:8002)

---

## 3. Guide d'Appel des API pour les Développeurs Backend

### A. Découper un enregistrement audio (`POST /vad/split`)
```bash
curl -X POST "http://localhost:8001/vad/split" \
  -H "accept: application/json" \
  -F "file=@mon_audio.mp3"
```
*Retourne la liste des portions de parole avec des URLs sécurisées `/audio/segments/{session_id}/{filename}` pour chaque extrait WAV.*

---

### B. Transcrire de l'audio (`POST /transcribe`)
```bash
curl -X POST "http://localhost:8001/transcribe" \
  -F "file=@client_vocal.wav" \
  -F "language=fr"
```
*Exemple de réponse JSON reçue :*
```json
{
  "text": "Je veux envoyer de l'argent à maman en fcfa",
  "language": "fr",
  "duration": 4.12,
  "segments": [...]
}
```

---

### C. Comprendre et extraire les entités (`POST /comprehension/analyze`)
Le backend transmet le texte reçu du Service Voix au Service Compréhension :
```bash
curl -X POST "http://localhost:8002/comprehension/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Je veux envoyer de l'\''argent à maman en fcfa",
    "language": "fr",
    "expected_fields": ["intent", "destinataire", "devise"]
  }'
```
*Exemple de réponse structurée reçue (Zéro appel LLM si présent dans le dictionnaire) :*
```json
{
  "language": "fr",
  "raw_text": "Je veux envoyer de l'argent à maman en fcfa",
  "intent": {
    "field_name": "intent",
    "value": "transfert_argent",
    "score": 100.0,
    "source": "dictionary"
  },
  "entities": {
    "destinataire": {
      "field_name": "destinataire",
      "value": "maman",
      "score": 100.0,
      "source": "dictionary"
    },
    "devise": {
      "field_name": "devise",
      "value": "fcfa",
      "score": 100.0,
      "source": "dictionary"
    }
  },
  "available_fields": ["intent", "destinataire", "devise"],
  "missing_fields": [],
  "llm_used": false,
  "is_complete": true
}
```

---

### D. Synthétiser une réponse vocale (`POST /synthesize`)
Pour renvoyer un message audio à l'utilisateur :
```bash
curl -X POST "http://localhost:8001/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Akwɛ́ sɛ́dó nɔ̂ en fcfa",
    "language": "fon"
  }' \
  --output reponse_fon.wav
```

---

## 4. Informations Requises de l'Équipe Plateforme

Pour finaliser l'intégration réseau et métier avec le backend principal, l'équipe plateforme doit clarifier les 7 points suivants :

1. **Langage et framework du backend principal :**  
   *(Ex: Node.js/NestJS, Spring Boot, Laravel, Django, Go)* afin de fournir si besoin un client SDK pré-généré.
2. **Format et transport des fichiers audio :**  
   Quels codecs sont produits par l'application client (WAV, MP3, WebM/Opus, M4A) ? Les requêtes arrivent-elles en blocs de parole complets ou un flux continu (WebSockets) est-il envisagé ?
3. **Mécanisme d'authentification interservices :**  
   Quel modèle de sécurité doit être activé (header `X-API-Key`, token JWT Bearer, ou réseau privé interne Docker/Kubernetes) ?
4. **Dictionnaire métier réel :**  
   La liste exacte des intentions et entités réelles de l'application (types d'opérations, devises, villes et localités, noms d'agences) doit être déposée dans `comprehension_service/data/dictionaries/{fr,fon,yor}.json`.
5. **Fournisseur et modèle LLM :**  
   Quelle solution d'inférence LLM sera utilisée en production pour compléter les champs manquants : OpenAI API (clé publique), Azure OpenAI, ou un modèle open-source hébergé localement (vLLM / Ollama) pour préserver la souveraineté des données ?
6. **Infrastructure de déploiement :**  
   Les conteneurs seront-ils déployés sur une VM unique avec Docker Compose ou orchestrés sur Kubernetes ?
7. **Ressources CPU / GPU allouées :**  
   L'infrastructure met-elle à disposition un GPU Nvidia pour le service voix afin d'accélérer le temps de traitement ?

---

## 5. Note Juridique sur les Licences des Modèles MMS

* Les modèles de transcription (`facebook/mms-1b-all`) et de synthèse vocale (`facebook/mms-tts-*`) publiés par Meta AI sont distribués sous licence **CC-BY-NC 4.0** (Creative Commons Non-Commercial).
* Si la plateforme collective vise une exploitation commerciale directe, une revue de licence ou l'usage d'adaptateurs autorisés devra être validée avec le responsable juridique du projet.
