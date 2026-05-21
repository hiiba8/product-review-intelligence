import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

os.environ["SERPAPI_API_KEY"] = ""

from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from serpapi import GoogleSearch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("sentiment_model_3classes")
dl_model = DistilBertForSequenceClassification.from_pretrained("sentiment_model_3classes")
dl_model.eval()
print("Modèle chargé")

llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

logger = logging.getLogger("crew_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.FileHandler("agent_log.json", encoding="utf-8"))

def log_action(agent, action, result, status="success"):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "status": status,
        "result": str(result)[:500]
    }
    logger.info(json.dumps(entry, ensure_ascii=False))

@tool("sentiment_analysis_tool")
def sentiment_analysis_tool(reviews: str) -> str:
    """
    Analyse le sentiment d'avis produit séparés par '|'.
    Retourne un JSON avec label et score de confiance par avis.
    Input : textes séparés par '|'.
    Output : JSON structuré.
    """
    try:
        texts = [t.strip() for t in reviews.split("|") if t.strip()]
        if not texts:
            return "Erreur : aucun texte fourni."
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        results = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = dl_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            results.append({
                "text": text[:100],
                "sentiment": label_map.get(pred_idx, str(pred_idx)),
                "confidence": round(probs[pred_idx].item(), 3)
            })
        sentiments = [r["sentiment"] for r in results]
        summary = {
            "total_reviews": len(results),
            "breakdown": {s: sentiments.count(s) for s in set(sentiments)},
            "details": results
        }
        output = json.dumps(summary, ensure_ascii=False, indent=2)
        log_action("SentimentAnalyst", f"analyze ({len(texts)} reviews)", output)
        return output
    except Exception as e:
        error_msg = f"Erreur sentiment_analysis_tool : {e}"
        log_action("SentimentAnalyst", "analyze", error_msg, status="error")
        return error_msg

@tool("web_search_reviews_tool")
def web_search_reviews_tool(product: str) -> str:
    """
    Recherche des avis clients en ligne pour un produit via SerpApi.
    Input : nom du produit.
    Output : snippets d'avis trouvés sur le web.
    """
    try:
        snippets = []
        search = GoogleSearch({
            "q": f"{product} customer reviews pros cons",
            "api_key": os.environ["SERPAPI_API_KEY"],
            "num": 6
        })
        results = search.get_dict().get("organic_results", [])
        for r in results:
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            if snippet:
                snippets.append(f"[{title}] {snippet}")
        if not snippets:
            return "Aucun avis trouvé."
        output = "\n\n".join(snippets[:5])
        log_action("MarketResearcher", f"search: {product}", f"{len(snippets)} résultats")
        return output
    except Exception as e:
        error_msg = f"Erreur web_search_reviews_tool : {e}"
        log_action("MarketResearcher", f"search: {product}", error_msg, status="error")
        return error_msg

sentiment_agent = Agent(
    role="Sentiment Analyst",
    goal="Analyser le sentiment des avis clients via sentiment_analysis_tool.",
    backstory="Expert NLP utilisant DistilBERT fine-tuné sur Amazon Reviews.",
    tools=[sentiment_analysis_tool],
    llm=llm,
    verbose=True,
    max_iter=4,
)

market_agent = Agent(
    role="Market Researcher",
    goal="Rechercher des avis en ligne et identifier forces et faiblesses du produit.",
    backstory="Analyste marché spécialisé en intelligence produit.",
    tools=[web_search_reviews_tool],
    llm=llm,
    verbose=True,
    max_iter=4,
)

report_agent = Agent(
    role="Report Generator",
    goal="Synthétiser les analyses en rapport structuré professionnel.",
    backstory="Consultant senior en intelligence produit.",
    tools=[],
    llm=llm,
    verbose=True,
    max_iter=4,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    product: str

# ── Analyse sentiment directe (pour les stats) ────────────────────────────────
def run_sentiment_direct(texts: list[str]) -> dict:
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = dl_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        results.append({
            "text": text[:100],
            "sentiment": label_map.get(pred_idx, str(pred_idx)),
            "confidence": round(probs[pred_idx].item(), 3)
        })
    sentiments = [r["sentiment"] for r in results]
    breakdown = {s: sentiments.count(s) for s in ["positive", "neutral", "negative"]}
    avg_conf = round(sum(r["confidence"] for r in results) / len(results), 3) if results else 0
    dominant = max(breakdown, key=breakdown.get) if results else "neutral"
    return {
        "total_reviews": len(results),
        "breakdown": breakdown,
        "avg_confidence": avg_conf,
        "dominant_sentiment": dominant,
        "details": results
    }

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    product = request.product

    try:
        raw_reviews = []
        search = GoogleSearch({
            "q": f"{product} customer reviews",
            "api_key": os.environ["SERPAPI_API_KEY"],
            "num": 8
        })
        results = search.get_dict().get("organic_results", [])
        for r in results:
            snippet = r.get("snippet", "")
            if snippet and len(snippet) > 30:
                raw_reviews.append(snippet[:200])

        if not raw_reviews:
            return {
                "status": "error",
                "message": f"Aucun avis trouvé pour '{product}'. Essaie un nom plus précis."
            }

        # ── Stats sentiment directes ──────────────────────────────────────────
        sentiment_stats = run_sentiment_direct([r[:150] for r in raw_reviews[:6]])

        reviews_str = " | ".join([r[:80] for r in raw_reviews[:3]])
        log_action("Orchestrator", "crew_kickoff", f"Produit: {product}")

        task_sentiment = Task(
            description=f"""Tu DOIS appeler sentiment_analysis_tool avec exactement ce texte :
    '{reviews_str}'
Ne génère pas de résultat toi-même. Utilise l'outil.""",
            expected_output="JSON avec breakdown et details.",
            agent=sentiment_agent,
        )

        task_research = Task(
            description=f"Recherche '{product}' via web_search_reviews_tool. Donne 3 forces et 3 faiblesses.",
            expected_output="3 forces, 3 faiblesses.",
            agent=market_agent,
        )

        task_report = Task(
            description=f"Rapport pour '{product}': 1.Résumé 2.Sentiment (sans mentionner les scores de confiance) 3.Marché 4.Recommandations 5.Conclusion",
            expected_output="Rapport en 5 sections.",
            agent=report_agent,
            context=[task_sentiment, task_research],
        )

        crew = Crew(
            agents=[sentiment_agent, market_agent, report_agent],
            tasks=[task_sentiment, task_research, task_report],
            process=Process.sequential,
            verbose=False,
            step_callback=lambda x: __import__('time').sleep(3)
        )

        import time
        for attempt in range(3):
            try:
                result = crew.kickoff()
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "RateLimitError" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"Rate limit, attente {wait}s...")
                    time.sleep(wait)
                else:
                    raise e

        log_action("Orchestrator", "crew_finished", str(result)[:500])
        import re
        report_text = str(result)
        report_text = re.sub(r'\$\\boxed\{', '', report_text)
        report_text = re.sub(r'\}\$', '', report_text)
        report_text = re.sub(r'The final answer is:', '', report_text)
        report_text = report_text.strip()

        return {
            "status": "success",
            "product": product,
            "reviews_analyzed": raw_reviews[:6],
            "sentiment_stats": sentiment_stats,
            "report": report_text
        }

    except Exception as e:
        log_action("Orchestrator", "crew_error", str(e), status="error")
        return {"status": "error", "message": str(e)}