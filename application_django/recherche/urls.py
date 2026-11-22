from django.urls import path

from . import views

urlpatterns = [
    path("", views.accueil, name="accueil"),
    path("methode/", views.methode, name="methode"),
    path("resultats/", views.resultats, name="resultats"),
    path("corpus/", views.corpus, name="corpus"),
    path("api/recherche/", views.api_recherche, name="api_recherche"),
]
