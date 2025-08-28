from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ha, name='ha'),
    path('model/', views.model, name='model'),
    path('dash/', views.dash, name='dash')
]