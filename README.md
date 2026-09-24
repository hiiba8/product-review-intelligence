# Product Review Intelligence

A multi-agent system for analyzing product reviews and generating structured market insights.

Enter a product name, collect online reviews, analyze their sentiment with a fine-tuned DistilBERT model, and generate a structured report.

---

## How it works

The system is built around three specialized agents:

<table>
<tr>
<th>Agent</th>
<th>Role</th>
</tr>
<tr>
<td><strong>Market Researcher</strong></td>
<td>Collects product reviews from Google using SerpApi.</td>
</tr>
<tr>
<td><strong>Sentiment Analyst</strong></td>
<td>Classifies reviews as negative, neutral, or positive using a fine-tuned DistilBERT model.</td>
</tr>
<tr>
<td><strong>Report Generator</strong></td>
<td>Combines the results into a structured market report.</td>
</tr>
</table>

The final report is organized into:

**Summary · Sentiment · Market Insights · Recommendations · Conclusion**

---

## Architecture

```text
                    Product name
                         |
                         v
                Market Researcher
                         |
                         v
                  Online reviews
                         |
                         v
                 Sentiment Analyst
                         |
                         v
                Fine-tuned DistilBERT
                         |
                         v
                 Report Generator
                         |
                         v
                Market Report
```

---

## Technology Stack

<table>
<tr>
<td><strong>Multi-agent system</strong></td>
<td>CrewAI</td>
</tr>
<tr>
<td><strong>Language model</strong></td>
<td>Ollama · Llama 3</td>
</tr>
<tr>
<td><strong>Sentiment analysis</strong></td>
<td>DistilBERT · Hugging Face</td>
</tr>
<tr>
<td><strong>Web search</strong></td>
<td>SerpApi</td>
</tr>
<tr>
<td><strong>Backend</strong></td>
<td>FastAPI</td>
</tr>
<tr>
<td><strong>Frontend</strong></td>
<td>HTML · CSS · JavaScript</td>
</tr>
</table>

---

## Sentiment Model

The sentiment classifier was fine-tuned on a balanced subset of the Amazon Reviews dataset.

**Training setup**

* 7,461 examples
* 3 sentiment classes
* 2 epochs
* NVIDIA T4 GPU

### Results

| Class    | F1-score |
| -------- | -------: |
| Negative |     0.73 |
| Neutral  |     0.66 |
| Positive |     0.82 |

---

## Project Structure

```text
.
├── main.py
├── index.html
├── Notebook1.ipynb
└── README.md
```

---

## Run locally

### Requirements

* Python 3.10+
* Ollama with Llama 3
* SerpApi API key

### Installation

```bash
pip install -r requirements.txt
ollama pull llama3
```

Configure your SerpApi API key and start the application:

```bash
uvicorn main:app --reload
```

Then open the frontend and enter a product name to start the analysis.

---

## Academic Project

Developed as part of the Deep Learning course at UIR ESIN, 2025–2026.
