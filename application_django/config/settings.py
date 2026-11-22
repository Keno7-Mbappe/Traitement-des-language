"""Réglages Django du prototype de recherche sémantique (projet 8).

Application de démonstration : pas de base de données, pas de comptes utilisateurs.
Deux situations :
  - sur l'ordinateur (par défaut) : mode développement, clé secrète d'exemple, `python manage.py runserver` ;
  - en ligne sur Render (la variable RENDER est définie par la plateforme) : DEBUG coupé, clé secrète lue dans
    DJANGO_SECRET_KEY, fichiers statiques servis par WhiteNoise, adresse du site acceptée automatiquement.
"""
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent
EN_LIGNE = bool(os.environ.get("RENDER"))          # vrai sur Render

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if EN_LIGNE:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY doit être définie sur la plateforme d'hébergement.")
    SECRET_KEY = "projet8-demonstration-pas-pour-la-production"          # exemple : uniquement pour le développement

DEBUG = os.environ.get("DJANGO_DEBUG", "0" if EN_LIGNE else "1") == "1"

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
ALLOWED_HOSTS += [h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",") if h.strip()]
if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
    ALLOWED_HOSTS.append(os.environ["RENDER_EXTERNAL_HOSTNAME"])

INSTALLED_APPS = ["django.contrib.staticfiles", "recherche"]
MIDDLEWARE = ["django.middleware.common.CommonMiddleware"]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request",
                                           "recherche.contexte.contexte_global"]},
    }
]

DATABASES = {}          # aucune base : les données sont dans le dossier artefacts/

LANGUAGE_CODE = "fr"
TIME_ZONE = "Africa/Djibouti"
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"          # feuilles de style, scripts et polices : recherche/static/recherche/
STATIC_ROOT = BASE_DIR / "staticfiles"          # rempli par « python manage.py collectstatic » au déploiement

if EN_LIGNE:
    MIDDLEWARE = [
        "django.middleware.security.SecurityMiddleware",
        "whitenoise.middleware.WhiteNoiseMiddleware",          # sert les fichiers statiques sans serveur web séparé
        "django.middleware.common.CommonMiddleware",
        "django.middleware.csrf.CsrfViewMiddleware",
        "django.middleware.clickjacking.XFrameOptionsMiddleware",
    ]
    STORAGES = {
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
    }
    WHITENOISE_MAX_AGE = 3600
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")          # Render termine le HTTPS devant l'application
    SECURE_HSTS_SECONDS = 3600          # le navigateur ne revient pas en HTTP pendant une heure (prudent : à augmenter plus tard)
    X_FRAME_OPTIONS = "DENY"
    SECURE_REFERRER_POLICY = "same-origin"
