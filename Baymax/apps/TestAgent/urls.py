# urls.py
from django.urls import path
from . import views

app_name = 'TestAgent'

urlpatterns = [
    path('', views.upload_file, name='upload'),  # Name is 'upload'
    path('ask-question/', views.askquestion, name='askquestion'), # Name is 'ask'
    # Add path for clear_table if you have it
    path('clear/', views.clear_table, name='clear_table'),
]
