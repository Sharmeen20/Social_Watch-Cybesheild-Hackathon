from sentence_transformers import SentenceTransformer, util
from django.conf import settings
from transformers import pipeline
import os
import pickle 
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
import yake

'''
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, "ml_model", "sentiment.pkl") 

with open(MODEL_FILE, "rb") as f:
    sentiment_model  = pickle.load(f)  
'''

import gdown
import joblib
import os

# Google Drive file ID
file_id = "1InGNz1xmSiOghPaUN7rbn6WxNV67CWt6"
url = f"https://drive.google.com/uc?id={file_id}"

# Local path to save the model
local_model_path = "sentiment.pkl"

# Download if not already downloaded
if not os.path.exists(local_model_path):
    gdown.download(url, local_model_path, quiet=False)

# Load the ML model
sentiment_model = joblib.load(local_model_path)

def senty_pred(text):
    results = sentiment_model (text)
    return results[0]['label']

'''
model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
'''

# Google Drive file ID
file_id = "1JwW31HDNuopyAnGr0MbfFbmFuy710vgt"
url = f"https://drive.google.com/uc?id={file_id}"

# Local path to save the model
local_model_path = "model_NLP.pkl"

# Download if not already downloaded
if not os.path.exists(local_model_path):
    gdown.download(url, local_model_path, quiet=False)

# Load the ML model
model = joblib.load(local_model_path)

# Google Drive file ID
file_id = "1bz8EFOE5Uk-FzLDR0qYpsGRwHoGuVB_B"
url = f"https://drive.google.com/uc?id={file_id}"

# Local path to save the tokenizer
local_tokenizer_path = "tokenizer.pkl"

# Download if not already downloaded
if not os.path.exists(local_tokenizer_path):
    gdown.download(url, local_tokenizer_path, quiet=False)

# Load the tokenizer
tokenizer = joblib.load(local_tokenizer_path)

SERPER_API_KEY = "cb0928906e7b91c67200cfd5cb820fc8b2b9a371"
SERPER_NEWS_URL = "https://google.serper.dev/news"

headers = {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json"
}
def extract_keywords_yake(text, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(top=max_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def search_news_with_serper(keywords):
    query = " ".join(keywords)
    payload = {"q": query, "num": 5}
    response = requests.post(SERPER_NEWS_URL, headers=headers, json=payload)
    if response.status_code == 200:
        news_items = response.json().get("news", [])
        return news_items
    else:
        print("Serper API Error:", response.status_code, response.text)
        return []
def analyze_stances(input_text, fetched_news):
    stances = []
    for news in fetched_news:
        snippet = news.get("snippet", "")
        if snippet == "":
            continue
        if input_text.lower() in snippet.lower():
            stances.append("agree")
        elif any(word in snippet.lower() for word in input_text.lower().split()):
            stances.append("discuss")
        else:
            stances.append("unrelated")
    return stances
def evaluate_news(input_text, stances):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    fake_real_label = "Fake" if predicted_class == 1 else "Real"

    if "agree" in stances:
        stance_label = "Real"
    elif "discuss" in stances:
        stance_label = "Neutral"
    elif "unrelated" in stances:
        stance_label = "Neutral"
    else:
        stance_label = "Fake"

    if fake_real_label == "Fake" and stance_label != "Real":
        return "Fake"
    elif fake_real_label == "Real" and stance_label == "Neutral":
        return "Neutral"
    else:
        return "Real"
    
def verify_news(news_text):
    keywords = extract_keywords_yake(news_text)
    fetched_news = search_news_with_serper(keywords)
    stances = analyze_stances(news_text, fetched_news)
    result = evaluate_news(news_text, stances)
    return result