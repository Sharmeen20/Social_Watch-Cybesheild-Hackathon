from django.shortcuts import render
from .forms import PredictionForm
from django.conf import settings
import pandas as pd
import os
from ml_model.sentiment_model import senty_pred, verify_news
def load_csv(file_name):
    """Utility to load CSV from ml_model directory."""
    path = os.path.join(settings.BASE_DIR, 'ml_model', file_name)
    return pd.read_csv(path)

def ha(request):
    """
    Home view for displaying threat detection and sentiment analysis.
    Loads data, identifies threats based on keywords, and summarizes sentiment.
    """
    try:
        df = load_csv('500.csv')
        df['Sentiment'] = df['Sentiment'].fillna('No Data')

        threat_keywords = [
            'attack', 'murder', 'rape', 'assault', 'kidnap', 'violence',
            'bomb', 'shoot', 'stab', 'robbery', 'terror', 'threat', 'kill'
        ]

        # Flag threats based on keyword presence
        df['Threat'] = df['Text'].apply(
            lambda x: 1 if any(word in str(x).lower() for word in threat_keywords) else 0
        )

        total_posts = len(df)
        threats_detected = df['Threat'].sum()
        overall_sentiment = df['Sentiment'].mode().iloc[0] if not df.empty else "No Data"

        keyword_counts = {
            kw: df['Text'].str.lower().str.contains(kw).sum() for kw in threat_keywords
        }

        context = {
            "total_posts": total_posts,
            "threats_detected": threats_detected,
            "overall_sentiment": overall_sentiment,
            "threat_keywords": threat_keywords,
            "df": df,
            "keyword_counts": keyword_counts,
        }
        return render(request, "index.html", context)

    except Exception as e:
        return render(request, "index.html", {"error": f"Error loading data: {e}"})


def model(request):
    """
    View for sentiment prediction and fake news verification.
    Uses form input to trigger ML model predictions.
    """
    prediction = None
    result = None

    form = PredictionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data['feature1']

        if "predict_sentiment" in request.POST:
            prediction = senty_pred(data)

        elif "check_result" in request.POST:
            result = verify_news(data)

    return render(request, "model.html", {
        "form": form,
        "prediction": prediction,
        "result": result,
    })
