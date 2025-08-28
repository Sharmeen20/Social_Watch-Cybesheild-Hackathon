from sentence_transformers import SentenceTransformer, util
from django.conf import settings
from transformers import pipeline

def senty_pred(text):
    sentiment_analyzer = pipeline("sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
                )
    results = sentiment_analyzer(text)
    return results[0]['label']
