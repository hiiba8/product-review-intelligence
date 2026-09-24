# Product Review Intelligence

A multi-agent system for analyzing product reviews and generating structured market insights.

The user enters a product name. The system collects online reviews, analyzes their sentiment using a fine-tuned DistilBERT model, and generates a structured market report.

## Overview

The system combines sentiment analysis, web data collection, and multi-agent orchestration into a single pipeline.

### Agents

| Agent                 | Role                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| **Sentiment Analyst** | Classifies reviews as negative, neutral, or positive using a fine-tuned DistilBERT model |
| **Market Researcher** | Collects product reviews from Google through SerpApi                                     |
| **Report Generator**  | Combines the collected information into a structured market report                       |

The generated report contains five sections:

**Summary · Sentiment Analysis · Market Insights · Recommendations · Conclusion**

## Architecture

```text
Product name
     │
     ▼
Market Researcher
     │
     ▼
Reviews collected through SerpApi
     │
     ▼
Sentiment Analyst
     │
     ▼
DistilBERT sentiment classification
     │
     ▼
Report Generator
     │
     ▼
Structured market report
```

## Technologies

| Component             | Technology              |
| --------------------- | ----------------------- |
| Multi-agent framework | CrewAI                  |
| LLM                   | Ollama / Llama 3        |
| Sentiment model       | Fine-tuned DistilBERT   |
| Web search            | SerpApi                 |
| Backend               | FastAPI                 |
| Frontend              | HTML · CSS · JavaScript |

## Sentiment Model

The sentiment classifier was fine-tuned on a balanced subset of the Amazon Reviews dataset.

* **7,461 training examples**
* **3 sentiment classes**
* **2 training epochs**
* **NVIDIA T4 GPU**

### Results

| Class    | F1-score |
| -------- | -------: |
| Negative |     0.73 |
| Neutral  |     0.66 |
| Positive |     0.82 |

The fine-tuned model is stored in `sentiment_model_3classes/`.

## Running the Project

### Requirements

* Python 3.10+
* Ollama
* Llama 3
* SerpApi API key

### Installation

```bash
pip install -r requirements.txt
ollama pull llama3
```

Configure your SerpApi API key, then start the FastAPI application:

```bash
uvicorn main:app --reload
```

Open the frontend and enter a product name to start the analysis.

## Project Context

Developed as part of the Deep Learning course at **UIR ESIN** during the 2025–2026 academic year.

