from django.urls import *
from . import views

urlpatterns = [
	path("", views.predict_crop, name="predict_crop_two_models")
]
