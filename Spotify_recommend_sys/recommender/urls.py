from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views
from .views import SignUpView, logout_view

urlpatterns = [
    # Existing URL patterns
    path('', views.home, name='home'),
    path('guest/', views.guest_page, name='guest_page'),
    path('initial_rating/', views.initial_rating, name='initial_rating'),
    path('start_recommendation/', views.start_recommendation, name='start_recommendation'),
    path('rate_recommendations/', views.rate_recommendations, name='rate_recommendations'),
    path('final_recommendations/', views.final_recommendations, name='final_recommendations'),
    path('user_history/', views.user_history, name='user_history'),
    
    # Authentication URL patterns
    path('accounts/login/', auth_views.LoginView.as_view(), name='login'),
    path('accounts/logout/', logout_view, name='logout'),
    path('accounts/signup/', SignUpView.as_view(), name='signup'),
    path('accounts/password_change/', auth_views.PasswordChangeView.as_view(), name='password_change'),
    path('accounts/password_change/done/', auth_views.PasswordChangeDoneView.as_view(), name='password_change_done'),
    path('accounts/password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('accounts/password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('accounts/reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('accounts/reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
]