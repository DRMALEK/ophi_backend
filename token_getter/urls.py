from django.contrib import admin
from django.urls import path, include
from instgram_crop import views as views2
from instgram_two_models import views

#views.load_model()
#views.load_model2()
views2.load_model()

urlpatterns = [
    path('admin/', admin.site.urls),
    path("auth/" , include("myapp.urls")),
    path("instgram_predict/", include("instgram_crop.urls")),
    path("instgram_two_models/", include("instgram_two_models.urls"))
]
