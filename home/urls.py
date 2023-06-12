from django.urls import path, include

from home import views

urlpatterns = [
    path('', views.home_index, name='home_index'),
    path('reinstall', views.model_reinstall, name='model_reinstall'),
    path('about', views.about_project, name='about_us'),

]
