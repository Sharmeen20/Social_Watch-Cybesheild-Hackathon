from django.shortcuts import render
from .forms import PredictionForm
import pandas as pd
from django.conf import settings
import plotly.express as px
import networkx as nx
from pyvis.network import Network
from networkx.algorithms import community
import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
from ml_model import visualizations as viz
from ml_model.sentiment_model import senty_pred, verify_news
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import base64
import io
from sklearn.preprocessing import LabelEncoder
import plotly.io as pio
import seaborn as sns

# def ha(request):
#    return render(request, "index.html")

def ha(request):
    csv_path = os.path.join(settings.BASE_DIR, 'ml_model','500.csv')    
    df = pd.read_csv(csv_path)
    df['Sentiment'] = df['Sentiment'].fillna('No Data')

    threat_keywords = [
        'attack', 'murder', 'rape', 'assault', 'kidnap', 'violence', 
        'bomb', 'shoot', 'stab', 'robbery', 'terror', 'threat', 'kill'
        ]

    df['Threat'] = df['Text'].apply(lambda x: 1 if any(word in str(x).lower() for word in threat_keywords) else 0)
    
    total_posts = len(df)
    threats_detected = df['Threat'].sum()
    overall_sentiment = df['Sentiment'].mode().iloc[0] if not df.empty else "No Data"

    keyword_counts = {}
    for kw in threat_keywords:
        keyword_counts[kw] = df['Text'].str.lower().str.contains(kw).sum()

    context = {
    "total_posts": total_posts,
    "threats_detected": threats_detected,
    "overall_sentiment": overall_sentiment,
    "threat_keywords": threat_keywords, 
    "df": df, 
    "keyword_counts": keyword_counts,
    }
    return render(request, "index.html", context)

def model(request):
    prediction = None
    result = None

    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data['feature1']

            # check which button was clicked
            if "predict_sentiment" in request.POST:
                prediction = senty_pred(data)

            elif "check_result" in request.POST:
                result = verify_news(data)

    else:
        form = PredictionForm()

    return render(request, "model.html", {
        "form": form,
        "prediction": prediction,
        "result": result,
    })



def dash(request):
    def load_data():
        csv_path = os.path.join(settings.BASE_DIR, 'ml_model','500.csv')
        data = pd.read_csv(csv_path)
        data.columns = data.columns.str.strip()
        data['Districts'] = data['Districts'].str.lower()
        return data

    data = load_data()

    def time_series_sentiment(data):
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        ts_data = data.dropna(subset=['Timestamp'])
        if 'SentimentScore' not in ts_data.columns:
            ts_data['SentimentScore'] = ts_data['Sentiment'].map({'Positive':1,'Neutral':0,'Negative':-1})
            time_series = ts_data.resample('D', on='Timestamp')['SentimentScore'].mean().reset_index()
            fig_ts = px.line(time_series, x='Timestamp', y='SentimentScore', title="Average Sentiment Score per Day")
            fig_ts.update_layout(xaxis_tickangle=45, margin=dict(l=40,r=40,t=50,b=50))
            return fig_ts.to_html(full_html=False)

    graph_time_series = time_series_sentiment(data)

    def overall_sentiment_pie(data):
        sentiment_counts = data['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment','Count']
        fig_pie = px.pie(
            sentiment_counts,
            names='Sentiment',
            values='Count',
            color='Sentiment',
            color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'},
            title="Overall Sentiment Distribution"
            )
        return fig_pie.to_html(full_html=False)

    graph_overall_pie = overall_sentiment_pie(data)

    selected_district = request.GET.get('district', 'all')
    metric = request.GET.get('metric', 'Negative')  # default = Negative

    def district_map(data, metric="Negative"):
        # Load GeoJSON
        geojson_path = os.path.join(settings.BASE_DIR, 'ml_model', 'MADHYA PRADESH_DISTRICTS.geojson')
        with open(geojson_path, 'r', encoding='utf-8') as f:
            mp_geojson = json.load(f)

        for feature in mp_geojson["features"]:
            feature["properties"]["dtname"] = feature["properties"]["dtname"].strip().lower()

        if metric in ["Negative", "Positive", "Neutral"]:
            filtered = data[data['Sentiment'] == metric]
            agg_data = filtered.groupby('Districts', as_index=False).agg({'Sentiment':'count'})
            agg_data.rename(columns={'Sentiment':'Count'}, inplace=True)
            colorscale = "Reds" if metric == "Negative" else ("Greens" if metric == "Positive" else "Greys")
            hover_text = f"{metric} Comments: "
        elif metric == "Likes":
            agg_data = data.groupby('Districts', as_index=False).agg({'Likes':'sum'})
            agg_data.rename(columns={'Likes':'Count'}, inplace=True)
            colorscale = "Viridis"
            hover_text = "Likes: "
        else:
            raise ValueError("Invalid metric. Choose 'Negative', 'Positive', 'Neutral', or 'Likes'.")

        # Merge with GeoJSON districts
        merged = pd.DataFrame({'Districts':[f['properties']['dtname'] for f in mp_geojson['features']]}).merge(
            agg_data, on="Districts", how="left").fillna(0)

        # Create map
        fig_map = go.Figure(go.Choroplethmapbox(
            geojson=mp_geojson,
            locations=merged['Districts'],
            z=merged['Count'],
            featureidkey="properties.dtname",
            colorscale=colorscale,
            marker_line_color="black",
            marker_line_width=1,
            hovertemplate=f"<b>%{{location}}</b><br>{hover_text}%{{z}}<extra></extra>"
            ))

        fig_map.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=5.5,
            mapbox_center={"lat":23.5, "lon":78.5},
            margin={"r":0,"t":50,"l":0,"b":0}
            )
        return fig_map.to_html(full_html=False)

    map_html = district_map(data, metric=metric)

    def district_charts(data, selected_district='all'):
        if selected_district != 'all':
            df_district = data[data['Districts']==selected_district]
            if df_district.empty:
                df_district = data
                selected_district = 'all'
        else:
            df_district = data

        # Bar Chart
        platform_sentiment = df_district.groupby(['Platform','Sentiment'])['Likes'].sum().reset_index()
        fig_bar = px.bar(
            platform_sentiment,
            x='Platform',
            y='Likes',
            color='Sentiment',
            barmode='group',
            title=f"Engagement by Platform and Sentiment for {selected_district.title()}"
            )
        graph_bar = fig_bar.to_html(full_html=False)

        # Pie Chart
        district_sentiment_counts = df_district['Sentiment'].value_counts().reset_index()
        district_sentiment_counts.columns = ['Sentiment','Count']
        fig_pie = px.pie(
            district_sentiment_counts,
            names='Sentiment',
            values='Count',
            color='Sentiment',
            color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'},
            title=f"Sentiment Distribution for {selected_district.title()}"
            )
        graph_pie = fig_pie.to_html(full_html=False)

        return graph_bar, graph_pie

    graph_district_bar, graph_district_pie = district_charts(data, selected_district)

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


def social_dashboard(request):
    # ----------------------------
    # Load dataset
    # ----------------------------
    csv_path = os.path.join(settings.BASE_DIR, 'ml_model', 'indian_social_media_dataset_with_influencers.csv')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # ----------------------------
    # 1️⃣ Engagement by Platform
    # ----------------------------
    metrics = df.groupby("platform")[["likes","shares","comments","impressions","reach"]].mean().reset_index()
    fig_engagement = px.bar(
        metrics,
        x="platform",
        y=["likes","shares","comments","impressions","reach"],
        barmode="group",
        title="Average Engagement Metrics by Platform"
    )
    engagement_html = fig_engagement.to_html(full_html=False)

    # ----------------------------
    # 2️⃣ Echo Chamber Analysis (with sentiment)
    # ----------------------------
    edges = []
    for val, group in df.groupby("audience_interests"):
        infls = group['influencer_id'].dropna().unique()
        for i in range(len(infls)):
            for j in range(i + 1, len(infls)):
                edges.append((infls[i], infls[j]))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    echo_img_path = None
    node_colors = []

    if len(G.nodes) > 0:
        # Map dominant sentiment per influencer
        user_sentiment = df.groupby('influencer_id')['sentiment'].agg(lambda x: x.value_counts().idxmax())
        color_map = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
        node_colors = [color_map.get(user_sentiment.get(node, 'neutral'), 'gray') for node in G.nodes()]

        # Layout and draw
        pos = nx.spring_layout(G, k=0.8, seed=42)
        plt.figure(figsize=(12, 10))
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1200,
            arrowsize=15,
            font_size=8,
            edge_color="black",
            width=0.5
        )
        plt.title("Influencer Polarization by Audience & Sentiment", fontsize=14)

        # Save to static folder
        echo_img_path = os.path.join(settings.BASE_DIR, "static/echo_chambers.png")
        plt.savefig(echo_img_path, bbox_inches='tight')
        plt.close()

    # ----------------------------
    # 3️⃣ Sentiment Over Time
    # ----------------------------
    sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
    df["sentiment_score"] = df["sentiment"].str.lower().map(sentiment_map)
    df["post_timestamp"] = pd.to_datetime(df["post_timestamp"], errors="coerce")
    sentiment_trend = df.groupby(pd.Grouper(key="post_timestamp", freq="W"))["sentiment_score"].mean().reset_index()
    
    fig_sentiment = px.line(
        sentiment_trend,
        x="post_timestamp",
        y="sentiment_score",
        title="Average Sentiment Score Over Time",
        markers=True
    )
    fig_sentiment.update_yaxes(title="Sentiment Score (-1=Neg, 0=Neu, 1=Pos)")
    sentiment_html = fig_sentiment.to_html(full_html=False)

    # ----------------------------
    # 4️⃣ Audience Age Groups
    # ----------------------------
    fig_age = px.histogram(
        df,
        x="age_group",
        color="sentiment",
        barmode="group",
        title="Sentiment Distribution by Age Group"
    )
    age_html = fig_age.to_html(full_html=False)

    # ----------------------------
    # 5️⃣ Mental Health / Wellbeing (simulated)
    # ----------------------------
    df["time_spent"] = (df["engagement_rate"] * 100 + df["reach"] / 1000).fillna(0) + np.random.normal(30, 5, len(df))
    df["wellbeing_score"] = (df["time_spent"] / df["time_spent"].max()) * 10 + np.random.normal(5, 2, len(df))
    
    fig_wellbeing = px.scatter(
        df,
        x="time_spent",
        y="wellbeing_score",
        color="platform",
        title="Time Spent vs Wellbeing Score by Platform",
        labels={"time_spent":"Time Spent (minutes)", "wellbeing_score":"Wellbeing Score"}
    )
    wellbeing_html = fig_wellbeing.to_html(full_html=False)

    # ----------------------------
    # Context & Render
    # ----------------------------
    context = {
        "engagement_html": engagement_html,
        "echo_img_path": echo_img_path,
        "sentiment_html": sentiment_html,
        "age_html": age_html,
        "wellbeing_html": wellbeing_html,
    }

    return render(request, "dash.html", context)

def random(request):
    # ----------------------------
    # Load dataset
    # ----------------------------
    csv_path = os.path.join(settings.BASE_DIR, 'ml_model', 'indian_social_media_dataset_with_influencers.csv')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    insights = {}

    # ----------------------------
    # 1️⃣ Conceptual Explanation
    # ----------------------------
    insights["understanding_algorithms"] = (
        "Social media algorithms decide which posts you see. "
        "Facebook/Instagram focus on engagement (likes, shares, comments), "
        "TikTok on watch-time, YouTube on recommendations & session duration, "
        "Twitter/X on recency & interactions."
    )

    # ----------------------------
    # 2️⃣ Feature Engineering & ML
    # ----------------------------
    features = ["likes", "comments", "shares", "impressions", "reach"]
    X = df[features].fillna(0)
    y = df["engagement_rate"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    insights["evaluation"] = f"RMSE = {rmse:.4f}, R² = {r2:.4f}"

    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    insights["feature_importance"] = importance.to_dict()

    # ----------------------------
    # 3️⃣ How Algorithms Learn & Adapt
    # ----------------------------
    df_aug = df.copy()
    df_aug["likes"] = df_aug["likes"] * 1.2  # simulate more likes
    X_aug = df_aug[features].fillna(0)
    y_aug = df_aug["engagement_rate"].fillna(0)

    model.fit(X_aug, y_aug)
    insights["adaptation"] = "Model retrained with boosted likes — shows how algorithms adapt with new feedback."

    # ----------------------------
    # 4️⃣ Comparisons & Insights
    # ----------------------------
    if "post_type" in df.columns and "sentiment" in df.columns and "platform" in df.columns:
        insights["comparisons"] = {
            "by_post_type": df.groupby("post_type")["engagement_rate"].mean().to_dict(),
            "sentiment_vs_engagement": df.groupby("sentiment")["engagement_rate"].mean().to_dict(),
            "by_platform": df.groupby("platform")["engagement_rate"].mean().to_dict(),
        }

    return render(request, "rm_model.html", {"insights": insights})





def render_chart(fig):
    """Convert Matplotlib figure to base64 image for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64

def charts_dashboard(request):
    # Load dataset
    csv_path = os.path.join(settings.BASE_DIR, 'ml_model', 'indian_social_media_dataset_with_influencers.csv')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # --------------------------
    # Chart 1: Engagement by Platform
    # --------------------------
    by_platform = df.groupby("platform")["engagement_rate"].mean().reset_index()
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.barplot(data=by_platform, x="platform", y="engagement_rate", ax=ax1)
    ax1.set_title("Engagement by Platform")
    ax1.set_ylabel("Avg Engagement Rate")
    chart1 = render_chart(fig1)

    # --------------------------
    # Chart 2: Sentiment vs Engagement
    # --------------------------
    sentiment_vs_engagement = df.groupby("sentiment")["engagement_rate"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.barplot(data=sentiment_vs_engagement, x="sentiment", y="engagement_rate", ax=ax2)
    ax2.set_title("Sentiment vs Engagement")
    ax2.set_ylabel("Avg Engagement Rate")
    chart2 = render_chart(fig2)

    # --------------------------
    # Chart 3: By Post Type
    # --------------------------
    by_post_type = df.groupby("post_type")["engagement_rate"].mean().reset_index()
    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.barplot(data=by_post_type, x="post_type", y="engagement_rate", ax=ax3)
    ax3.set_title("Engagement by Post Type")
    ax3.set_ylabel("Avg Engagement Rate")
    chart3 = render_chart(fig3)

    # --------------------------
    # Chart 4: Feature Importance (dummy example)
    # --------------------------
    importance = pd.DataFrame({
        "feature": ["likes", "comments", "shares", "reach", "impressions"],
        "score": [0.35, 0.25, 0.15, 0.15, 0.10]
    })
    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.barplot(data=importance, x="feature", y="score", ax=ax4)
    ax4.set_title("Feature Importance")
    chart4 = render_chart(fig4)

# --------------------------
# Chart 5: Top 5 Posts by ML Predicted Score
# --------------------------
    if "ml_predicted_score" in df.columns:
        top5_ml = df.nlargest(5, "ml_predicted_score")
        fig5, ax5 = plt.subplots(figsize=(6,4))
        sns.barplot(data=top5_ml, x="post_id", y="ml_predicted_score", ax=ax5)
        ax5.set_title("Top 5 Posts (ML Predicted Score)")
        chart5 = render_chart(fig5)
    else:
        chart5 = None

# --------------------------
# Chart 6: Top 5 Posts by Baseline Score
# --------------------------
    if "baseline_score" in df.columns:
        top5_base = df.nlargest(5, "baseline_score")
        fig6, ax6 = plt.subplots(figsize=(6,4))
        sns.barplot(data=top5_base, x="post_id", y="baseline_score", ax=ax6)
        ax6.set_title("Top 5 Posts (Baseline Score)")
        chart6 = render_chart(fig6)
    else:
        chart6 = None


    return render(request, "charts.html", {
        "chart1": chart1,
        "chart2": chart2,
        "chart3": chart3,
        "chart4": chart4,
        "chart5": chart5,
        "chart6": chart6,
    })


# real\Scripts\activate