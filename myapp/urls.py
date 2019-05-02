from django.urls import *
from . import views



urlpatterns = [
      path("create_key/<user_id>", views.create_key,name="create_key"),
      path("get_token/<user_id>", views.get_token ,name="create_token")
]
