from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('plot/', views.plot, name='plot'),
    path('comparaisonplot/', views.comparaisonplot, name='comparaisonplot'),
    path('about/', views.about, name='about'),
]