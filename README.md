# Django-based Sentiment Analysis and Social Media Dashboard

## Project Overview

This project is a Django web application that provides sentiment analysis and social media insights using Natural Language Processing (NLP) and Machine Learning (ML). It allows users to predict sentiment, verify news authenticity, analyze social media algorithms, and visualize engagement metrics through interactive dashboards.

---

## Features

* **Sentiment Prediction:** Predict sentiment (Positive, Neutral, Negative) from text input.
* **Fake News Verification:** Analyze news authenticity (Real, Fake, Neutral) using NLP models.
* **Social Media Insights:** Generate insights on engagement, echo chambers, sentiment trends, and feature importance.
* **District-Level Dashboards:** Visualize sentiment and engagement metrics across districts in Madhya Pradesh.
* **Static Charts:** Generate static charts using Matplotlib and Seaborn for further analysis.

---

## Project Structure

```
haga/                    # Project root
├── haga/                # Django project folder
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── sentiment/           # App folder
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   ├── tests.py
│   ├── ml_models/       # ML models, datasets, scripts
│   │   ├── 500.csv
│   │   ├── indian_social_media_dataset_with_influencers.csv
│   │   ├── MADHYA PRADESH_DISTRICTS.geojson
│   │   ├── sentiment_model.py
│   │   ├── visualizations.py
│   │   └── sentiment.pkl
│   ├── templates/       # App templates
│   │   ├── charts.html
│   │   ├── dash.html
│   │   ├── dashboard.html
│   │   ├── index.html
│   │   ├── model.html
│   │   └── rm_model.html
│   └── static/          # Static folder
│       └── echo_chambers.png
│
├── manage.py
├── requirements.txt
└── README.md
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/faizan1317/Social_Watch-Cybesheild-Hackathon.git
cd haga
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Apply migrations**

```bash
python manage.py migrate
```

5. **Run the development server**

```bash
python manage.py runserver
```

6. **Access the application**
   Open your browser and go to:

```
http://127.0.0.1:8000/
```

---

## URL Routing

URL Path      View Function           	Purpose                          	Template
/	             ha	            Threat detection & sentiment overview	        index.html
/model/	        model           Sentiment & fake news prediction            model.html
/dash/	         dash	       District-level sentiment dashboard          	dashboard.html
/dashboard/ social_dashboard	Influencer & engagement analysis         	dash.html
/random/	   random	      Algorithm insights & ML engagement simulation  rm_model.html
/charts_dashboard/ charts_dashboard	 Static charts (Matplotlib/Seaborn)	 charts.html

---

## ML & NLP Components

* **Sentiment Prediction**

  * `senty_pred(text)` in `ml_models/sentiment_model.py`
  * Predicts Positive, Neutral, Negative sentiment using a pre-trained model (`sentiment.pkl`)

* **Fake News Verification**

  * `verify_news(news_text)` in `ml_models/sentiment_model.py`
  * Keyword extraction using YAKE
  * Fetch news via Serper API
  * Evaluate using `bert-tiny-finetuned-fake-news-detection`

* **Visualizations**

  * `ml_models/visualizations.py`
  * Functions for choropleth maps, time series, pie charts, and district-level bar/pie charts

---

## Forms

* `PredictionForm` in `forms.py`

  * Single text input field for predicting sentiment

---

## How It Works

1. User accesses a URL
2. URL routing maps to the corresponding view
3. View processes data:

   * Uses ML models for predictions
   * Generates charts and visualizations
4. Context data is passed to the template
5. Template renders interactive dashboards and results

---

## Technologies Used

* **Backend:** Python, Django
* **Frontend:** HTML, CSS, JavaScript
* **Machine Learning:** Transformers, SentenceTransformer, pickle
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Data:** CSV datasets, GeoJSON for district maps

---
