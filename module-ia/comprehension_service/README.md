# Service Compréhension (`comprehension_service`) - Module IA

Le **Service Compréhension** de `module-ia` est le composant d'analyse sémantique et d'extraction d'informations structurées. Il reçoit les transcriptions générées par le service voix (ou des entrées textuelles directes) dans les trois langues officielles du projet (**Français**, **Fon**, **Yoruba**), les rapproche d'un dictionnaire métier par similarité lexicale (**RapidFuzz**), valide et structure les entités (**Pydantic**), et ne fait appel à un grand modèle de langage (**Instructor + LLM**) que de manière strictement conditionnelle pour compléter les champs obligatoires manquants.

---

## 1. Principes Fondamentaux & Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │ Transcription brute (JSON ou texte + langue) │
                  │     (Français 'fr', Fon 'fon', Yoruba 'yor')  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Normalisation Unicode & Diacritiques     │
                  │   (NFC, préservation stricte ɛ, ɔ, ɖ, ẹ, etc)│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        Rapprochement Lexical RapidFuzz       │
                  │     (Dictionnaire métier FR / FON / YOR)     │
                  │   - fuzz.ratio / partial_ratio / process     │
                  │   - seuil configurable (défaut : 80.0)       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Tous les champs requis sont résolus ? │
                     └───────────────────┬───────────────────┘
                                        / \
                                  OUI  /   \  NON (champs manquants)
                                      /     \
                                     /       ▼
                                    /     ┌────────────────────────────────────┐
                                    /     │  Appel Ciblé LLM via Instructor    │
                                    /     │  - Seuls les champs manquants sont │
                                    /     │    demandés au LLM                 │
                                    /     │  - Données dictionnaire préservées │
                                   │      └─────────────────┬──────────────────┘
                                   │                        │
                                   ▼                        ▼
                  ┌──────────────────────────────────────────────┐
                  │             Validation Pydantic              │
                  │   (Modèle standardisé ComprehensionOutput)   │
                  └──────────────────────────────────────────────┘
```

### Règle d'or : Appel conditionnel du LLM
- **Le LLM n'est JAMAIS appelé systématiquement.**
- Si toutes les informations requises (ex. `intent`, `destinataire`, `devise`) sont trouvées dans le dictionnaire avec un score $\ge$ au seuil configuré, le service répond **instantanément** sans aucun appel réseau vers le LLM (`llm_used: false`).
- Le LLM est sollicité **uniquement** si un ou plusieurs champs indispensables sont marqués comme manquants après le passage du dictionnaire.
- Les informations déjà résolues par le dictionnaire sont garanties et ne peuvent jamais être écrasées ou falsifiées par le LLM.
- En cas de panne, timeout ou désactivation du LLM, le système n'échoue pas : il retourne les informations certaines et signale proprement les champs manquants.

---

## 2. Technologies Utilisées

| Composant | Bibliothèque | Rôle |
| :--- | :--- | :--- |
| **Rapprochement lexical** | `rapidfuzz` | Calcul de distance d'édition, `fuzz.ratio`, `fuzz.partial_ratio`, `process.extract`, `process.extractOne` |
| **Validation & Typage** | `pydantic` | Typage strict, validation des entrées et sorties, métadonnées de traçabilité (`FieldSource`) |
| **Extraction LLM structurée** | `instructor` | Extraction garantie vers des schémas Pydantic typés depuis OpenAI ou API compatible |
| **Framework Web API** | `fastapi` | Point d'entrée HTTP `/comprehension/analyze` |

---

## 3. Prise en charge Linguistique & Diacritiques

Le système prend en charge trois langues de manière native :
- **Français (`fr`)** : accents standards (é, è, ê, à, etc.).
- **Fon (`fon`)** : voyelles ouvertes et consonnes spécifiques (`ɛ`, `ɔ`, `ɖ`, tons montants/descendants `´`, `` ` ``, `ˆ`, `ˇ`).
- **Yoruba (`yo` / `yor`)** : voyelles avec point souscrit (`ẹ`, `ọ`), consonne `ṣ`, tons aigus, graves et neutres (`á`, `à`, etc.).

La normalisation utilise `unicodedata.normalize("NFC", text)` et **ne supprime jamais les diacritiques indispensables** à la distinction des mots dans les langues béninoises et nigérianes.

---

## 4. Dictionnaires Métier

Les dictionnaires sont organisés sous `comprehension_service/data/dictionaries/` :
- `fr.json`
- `fon.json`
- `yor.json`

> **Note importante :** En l'absence de dictionnaire métier final fourni par le métier, des gabarits génériques et configurables sont fournis par défaut. Ces fichiers peuvent être enrichis ou remplacés sans modification de code.

---

## 5. Configuration (Variables d'environnement)

| Variable | Défaut | Description |
| :--- | :--- | :--- |
| `COMPREHENSION_FUZZY_MATCH_THRESHOLD` | `80.0` | Seuil minimal de similarité RapidFuzz (0 à 100) |
| `COMPREHENSION_FUZZY_PARTIAL_THRESHOLD` | `85.0` | Seuil pour les correspondances partielles |
| `DICTIONARIES_DIR` | `.../data/dictionaries` | Chemin vers les fichiers JSON de dictionnaires |
| `LLM_ENABLED` | `true` (si clé présente) | Active ou désactive les appels LLM complémentaires |
| `LLM_API_KEY` | `None` | Clé API pour le fournisseur LLM |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Modèle utilisé par Instructor |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | URL de base compatible OpenAI / vLLM / Ollama |
| `LLM_TIMEOUT_SECONDS` | `15.0` | Timeout maximal des requêtes LLM |

---

## 6. Structure des Fichiers

```
comprehension_service/
├── README.md
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration et seuils depuis l'environnement
│   ├── normalization.py       # Normalisation Unicode NFC et préservation diacritiques
│   ├── schemas.py             # Modèles Pydantic (FieldMatch, ComprehensionOutput, etc.)
│   ├── dictionary_matcher.py  # Moteur RapidFuzz (ratio, partial_ratio, extract)
│   ├── llm_service.py         # Client Instructor et prompts ciblés sur champs manquants
│   ├── comprehension.py       # Orchestrateur central du service compréhension
│   └── main.py                # Point d'entrée FastAPI
└── data/
    └── dictionaries/
        ├── fr.json            # Dictionnaire générique configurable FR
        ├── fon.json           # Dictionnaire générique configurable Fon
        └── yor.json           # Dictionnaire générique configurable Yoruba
```

---

## 7. Exécution des Tests

La suite de tests unitaires et d'intégration couvre l'ensemble des 15 exigences spécifiées :

```bash
# Exécution des tests du service compréhension
.venv/Scripts/pytest tests/comprehension -v

# Exécution de l'ensemble de la suite voix + compréhension
.venv/Scripts/pytest tests/voice -v
.venv/Scripts/pytest tests/comprehension -v
```
