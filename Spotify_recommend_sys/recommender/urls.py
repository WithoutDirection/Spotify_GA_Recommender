# recommender/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('initial-rating/', views.initial_rating, name='initial_rating'),
    path('start-recommendation/', views.start_recommendation, name='start_recommendation'),
    path('rate-recommendations/', views.rate_recommendations, name='rate_recommendations'),
    path('final-recommendations/', views.final_recommendations, name='final_recommendations'),
    path('history/', views.user_history, name='user_history'),
]
