# haga/ml_model/misinformation.py
import os, json, requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yake

HF_MODEL = os.getenv("FAKE_NEWS_MODEL", "mrm8488/bert-tiny-finetuned-fake-news-detection")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_NEWS_URL = "https://google.serper.dev/news"

# Optional headers only if key present
HEADERS = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"} if SERPER_API_KEY else None

# Load once at import
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
model.eval()
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

def extract_keywords(text: str, max_keywords: int = 5):
    kw = yake.KeywordExtractor(top=max_keywords)
    return [k for k, _score in kw.extract_keywords(text)]

def search_news(keywords):
    if not HEADERS:
        return []
    q = " ".join(keywords).strip()
    if not q:
        return []
    try:
        r = requests.post(SERPER_NEWS_URL, headers=HEADERS, json={"q": q, "num": 5}, timeout=10)
        r.raise_for_status()
        items = r.json().get("news", [])
    except Exception:
        return []
    # Keep only safe fields
    return [{
        "title": it.get("title"),
        "link": it.get("link"),
        "source": it.get("source"),
        "date": it.get("date"),
        "snippet": it.get("snippet", "")
    } for it in items]

def analyze_stances(input_text, news_list):
    t = (input_text or "").lower()
    stances = []
    for n in news_list:
        s = (n.get("snippet") or "").lower()
        if not s:
            continue
        if t and t in s:
            stances.append("agree")
        elif any(w and w in s for w in t.split()):
            stances.append("discuss")
        else:
            stances.append("unrelated")
    return stances

def classify_fake_real(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    label_id = torch.argmax(logits, dim=-1).item()
    # This model uses 1 => Fake, 0 => Real (as in your snippet)
    return "Fake" if label_id == 1 else "Real"

def fuse_labels(model_label, stances):
    if "agree" in stances:
        stance_label = "Real"
    elif "discuss" in stances or "unrelated" in stances:
        stance_label = "Neutral"
    else:
        stance_label = "Fake"

    if model_label == "Fake" and stance_label != "Real":
        final_label = "Fake"
    elif model_label == "Real" and stance_label == "Neutral":
        final_label = "Neutral"
    else:
        final_label = "Real"

    return final_label, stance_label

def verify_news(text: str):
    keywords = extract_keywords(text)
    news = search_news(keywords)
    stances = analyze_stances(text, news)
    model_label = classify_fake_real(text)
    final_label, stance_label = fuse_labels(model_label, stances)
    return {
        "final_label": final_label,
        "model_label": model_label,
        "stance_label": stance_label,
        "keywords": keywords,
        "news": news,
        "stances": stances,
    }
