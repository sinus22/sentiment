from django.urls import path, include

from home import views

urlpatterns = [
    path('', views.home_index, name='home_index'),
    path('test', views.home_test, name='home_test'),
    path('reinstall', views.model_reinstall, name='model_reinstall'),
    path('about', views.about_project, name='about_us'),

]
