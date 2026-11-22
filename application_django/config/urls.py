from django.urls import include, path

urlpatterns = [path("", include("recherche.urls"))]

handler404 = "recherche.views.page_introuvable"       # pages d'erreur au même style que le reste du site (quand DEBUG = False)
handler500 = "recherche.views.erreur_serveur"
