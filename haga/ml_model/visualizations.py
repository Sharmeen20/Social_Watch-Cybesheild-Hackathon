import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import os
from django.conf import settings

def load_data():
    csv_path = os.path.join(settings.BASE_DIR, 'ml_model', '500.csv')
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data['Districts'] = data['Districts'].str.lower()
    return data



def district_map(data, metric="Negative"):
    """
    Returns an HTML string of a choropleth map for Madhya Pradesh districts.
    
    metric: "Negative", "Positive", "Neutral", or "Likes"
    """
    # Load GeoJSON
    geojson_path = os.path.join(settings.BASE_DIR, 'ml_model', 'MADHYA PRADESH_DISTRICTS.geojson')
    with open(geojson_path, 'r', encoding='utf-8') as f:
        mp_geojson = json.load(f)

    for feature in mp_geojson["features"]:
        feature["properties"]["dtname"] = feature["properties"]["dtname"].strip().lower()

    # Prepare data based on metric
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


def time_series_sentiment(data):
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    ts_data = data.dropna(subset=['Timestamp'])
    if 'SentimentScore' not in ts_data.columns:
        ts_data['SentimentScore'] = ts_data['Sentiment'].map({'Positive':1,'Neutral':0,'Negative':-1})
    time_series = ts_data.resample('D', on='Timestamp')['SentimentScore'].mean().reset_index()
    fig_ts = px.line(time_series, x='Timestamp', y='SentimentScore', title="Average Sentiment Score per Day")
    fig_ts.update_layout(xaxis_tickangle=45, margin=dict(l=40,r=40,t=50,b=50))
    return fig_ts.to_html(full_html=False)

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
