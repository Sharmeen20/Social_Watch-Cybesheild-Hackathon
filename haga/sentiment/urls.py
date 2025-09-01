from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ha, name='ha'),
    path('model/', views.model, name='model'),
    path('dash/', views.dash, name='dash'),
    path('dashboard/', views.social_dashboard, name='dashboard'),
    path('random/', views.random, name='random'),
    path('charts_dashboard/', views.charts_dashboard, name='charts_dashboard')
]