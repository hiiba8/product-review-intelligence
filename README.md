# Product Review Intelligence

Système multi-agents d'analyse de sentiment pour les avis produits.

Entrez un nom de produit → le système collecte des avis en ligne, analyse leur sentiment avec un modèle DistilBERT fine-tuné, et génère un rapport de marché structuré.

## Stack

| Composant | Technologie |
|-----------|-------------|
| Agents | CrewAI (séquentiel) |
| LLM | Ollama (local, llama3) |
| Modèle DL | DistilBERT fine-tuné (73% accuracy, 3 classes) |
| Backend | FastAPI |
| Recherche web | SerpApi |
| Frontend | HTML / CSS / JS |

## Les trois agents

- **Sentiment Analyst** — classe chaque avis en négatif / neutre / positif via le modèle DistilBERT
- **Market Researcher** — récupère les avis Google via SerpApi
- **Report Generator** — rédige un rapport en 5 sections (Résumé, Sentiment, Marché, Recommandations, Conclusion)

## Lancer le projet

**Prérequis :** Python 3.10+, [Ollama](https://ollama.com) installé avec `llama3`, une clé SerpApi.

```bash
pip install -r requirements.txt
ollama pull llama3
uvicorn main:app --reload
```

Ouvrir `index.html` dans le navigateur, taper un nom de produit et cliquer sur **Lancer l'analyse**.

## Modèle

Entraîné sur Amazon Reviews (7 461 exemples équilibrés, 2 epochs, GPU T4) :

| Classe | F1 |
|--------|----|
| Négatif | 0.73 |
| Neutre | 0.66 |
| Positif | 0.82 |

Le modèle sauvegardé est dans `sentiment_model_3classes/`.

## Équipe

Aouami Salma · Zineb Arrami · Hiba Arbaoui · Ayyadi Marwa 
Encadrée par Pr. Hasna El Haji — UIR ESIN, Deep Learning S8 (2025–2026)
