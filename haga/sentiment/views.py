from django.shortcuts import render
from .forms import PredictionForm
import pandas as pd
from django.conf import settings

from ml_model import visualizations as viz

from ml_model.sentiment_model import senty_pred

def ha(request):
   return render(request, "index.html")

# def ha(request):
#     df = pd.read_csv(r"D:\hagothon\haga\ml_model\500.csv")
#     df['Sentiment'] = df['Sentiment'].fillna('No Data')

#     threat_keywords = [
#         'attack', 'murder', 'rape', 'assault', 'kidnap', 'violence', 
#         'bomb', 'shoot', 'stab', 'robbery', 'terror', 'threat', 'kill'
#         ]

#     df['Threat'] = df['Text'].apply(lambda x: 1 if any(word in str(x).lower() for word in threat_keywords) else 0)
    
#     total_posts = len(df)
#     threats_detected = df['Threat'].sum()
#     overall_sentiment = df['Sentiment'].mode().iloc[0] if not df.empty else "No Data"

#     keyword_counts = {}
#     for kw in threat_keywords:
#         keyword_counts[kw] = df['Text'].str.lower().str.contains(kw).sum()

#     context = {
#     "total_posts": total_posts,
#     "threats_detected": threats_detected,
#     "overall_sentiment": overall_sentiment,
#     "threat_keywords": threat_keywords, 
#     "df": df, 
#     "keyword_counts": keyword_counts,
#     }
#     return render(request, "index.html", context)

def model(request):
    prediction = None
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data['feature1'] 
            prediction = senty_pred(data)
    else:
        form = PredictionForm()
    return render(request, "model.html", {"prediction": prediction})

def dash(request):
    data = viz.load_data()
    graph_time_series = viz.time_series_sentiment(data)
    graph_overall_pie = viz.overall_sentiment_pie(data)

    selected_district = request.GET.get('district', 'all')
    metric = request.GET.get('metric', 'Negative')  # default = Negative

    map_html = viz.district_map(data, metric=metric)

    graph_district_bar, graph_district_pie = viz.district_charts(data, selected_district)

    districts_list = ['all'] + sorted(data['Districts'].unique())

    return render(request, "dashboard.html", {
        "map_html": map_html,
        "graph_time_series": graph_time_series,
        "graph_overall_pie": graph_overall_pie,
        "graph_district_bar": graph_district_bar,
        "graph_district_pie": graph_district_pie,
        "districts_list": districts_list,
        "selected_district": selected_district
    })

# real\Scripts\activate