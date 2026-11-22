"""À lancer UNE FOIS, avec internet, avant la démonstration. Au choix, un des deux services :

  - le serveur Colab du projet (notebook Serveur_Embeddings_Colab.ipynb), qui affiche les deux lignes à copier :
        $env:PROJET8_SERVEUR = "https://....trycloudflare.com"   (PowerShell)     set PROJET8_SERVEUR=...   (cmd)
        $env:PROJET8_CLE = "la clé affichée par Colab"
  - le service d'inférence de Hugging Face :
        $env:HF_TOKEN = "votre_jeton"   (PowerShell)     set HF_TOKEN=votre_jeton   (cmd)
  puis :
        python manage.py preparer_en_ligne                 (les 6 questions d'exemple)
        python manage.py preparer_en_ligne --evaluation    (en plus : les 83 requêtes du jeu d'évaluation)

La commande :
  1. vérifie que le service renvoie bien les MÊMES vecteurs que ceux du notebook (on calcule en ligne le vecteur
     de quelques passages et on le compare à celui enregistré dans le dossier artefacts/) ;
  2. SI ET SEULEMENT SI cette vérification réussit pour un modèle, calcule les vecteurs des questions et les garde dans
     un cache : ces questions marcheront ensuite sans aucun service (même sur le site en ligne), et le fichier
     questions_pretes.json garde la liste de celles qui sont prêtes. Un service infidèle ne remplit donc jamais le cache.
Le jeton et la clé ne sont jamais affichés ni enregistrés. Seul le texte des passages et des questions est envoyé au service."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from recherche import moteur, views

SEUIL_COSINUS = 0.99          # en dessous, le service ne renvoie pas le même vecteur que le notebook
PASSAGES_TESTES = (0, 57, 150)
FICHIER_REQUETES = moteur.DOSSIER_ARTEFACTS.parent.parent / "data" / "requetes.csv"
LANGUES_EXEMPLES = ("fr", "en", "so")          # langue des trois exemples « qui marchent » et des trois « cas difficiles »


class Command(BaseCommand):
    help = "Vérifie le service d'embeddings en ligne (serveur Colab ou Hugging Face) et prépare le cache des questions."

    def add_arguments(self, parser):
        parser.add_argument("--methodes", nargs="+", default=list(moteur.MODELES), choices=list(moteur.MODELES),
                            help="méthodes à embeddings à vérifier (par défaut : toutes)")
        parser.add_argument("--evaluation", action="store_true",
                            help="met aussi en cache les requêtes du jeu d'évaluation (data/requetes.csv)")
        parser.add_argument("--fichier-requetes", default=str(FICHIER_REQUETES),
                            help="fichier CSV des requêtes (colonnes requete, langue, type)")

    def questions_a_preparer(self, options):
        """[{requete, langue, type}] : les 6 exemples, puis (avec --evaluation) les requêtes d'évaluation, sans doublon."""
        questions, vues = [], set()

        def ajouter(texte, langue, genre):
            propre = moteur.nettoyer(texte)
            if propre and propre not in vues:
                vues.add(propre)
                questions.append({"requete": propre, "langue": langue, "type": genre})

        for liste, genre in ((views.EXEMPLES_QUI_MARCHENT, "exemple"), (views.EXEMPLES_DIFFICILES, "cas difficile")):
            for texte, langue in zip(liste, LANGUES_EXEMPLES):
                ajouter(texte, langue, genre)
        if options["evaluation"]:
            chemin = Path(options["fichier_requetes"])
            if not chemin.exists():
                raise CommandError(f"Fichier des requêtes introuvable : {chemin}")
            tableau = pd.read_csv(chemin, encoding="utf-8-sig").fillna("")
            for _, ligne in tableau.iterrows():
                ajouter(ligne["requete"], ligne["langue"], "évaluation")
        return questions

    def handle(self, *args, **options):
        if not (moteur.adresse_serveur() or moteur.jeton_hf()):
            raise CommandError("Aucun service en ligne configuré. Définissez PROJET8_SERVEUR et PROJET8_CLE (serveur Colab : "
                               "lancez Serveur_Embeddings_Colab.ipynb, il affiche les commandes) ou HF_TOKEN (jeton gratuit "
                               "sur huggingface.co > Settings > Access Tokens), puis relancez la commande.")
        service = {"serveur": "serveur Colab", "huggingface": "Hugging Face", None: "aucun (PROJET8_MODE ?)"}[moteur.service_en_ligne()]
        self.stdout.write(f"Service utilisé : {service}")
        passages = moteur.charger_passages()
        questions = self.questions_a_preparer(options)
        tout_va_bien = True

        for methode in options["methodes"]:
            nom_modele, _, prefixe_passage = moteur.MODELES[methode]
            self.stdout.write(f"\n=== {moteur.METHODES[methode]} ({nom_modele}) ===")
            matrice = moteur.matrice_dense(methode)

            self.stdout.write("1) vecteurs des passages : service comparé au notebook")
            fidele = True
            for i in PASSAGES_TESTES:
                try:
                    v = moteur.vecteur_service(methode, prefixe_passage + passages.iloc[i]["texte"])
                except moteur.MethodeIndisponible as raison:
                    self.stdout.write(self.style.ERROR(f"   ÉCHEC : {raison}"))
                    fidele = False
                    break
                cosinus = float(np.dot(v, matrice[i]) / np.linalg.norm(matrice[i]))
                etat = self.style.SUCCESS("OK") if cosinus >= SEUIL_COSINUS else self.style.ERROR("ÉCART")
                self.stdout.write(f"   passage {passages.iloc[i]['id']} : cosinus = {cosinus:.4f}  {etat}")
                fidele &= cosinus >= SEUIL_COSINUS
            if not fidele:
                tout_va_bien = False
                self.stdout.write(self.style.WARNING("2) questions : NON mises en cache (le service n'est pas fidèle au notebook)"))
                continue

            self.stdout.write(f"2) {len(questions)} question(s) : calcul et mise en cache")
            reussies = 0
            for q in questions:
                try:
                    moteur.vecteur_requete.cache_clear()
                    moteur.vecteur_requete(methode, q["requete"])
                    reussies += 1
                except moteur.MethodeIndisponible as raison:
                    self.stdout.write(self.style.ERROR(f"   {q['requete'][:60]} : {raison}"))
                    tout_va_bien = False
            self.stdout.write(f"   {reussies}/{len(questions)} question(s) en cache pour ce modèle  "
                              f"{self.style.SUCCESS('OK') if reussies == len(questions) else self.style.ERROR('INCOMPLET')}")

        # liste des questions prêtes : seulement celles dont les vecteurs sont en cache pour les modèles demandés
        cache = moteur.lire_cache()
        pretes = [q for q in questions if all(f"{m}|{q['requete']}" in cache for m in options["methodes"])]
        try:
            moteur.FICHIER_QUESTIONS.write_text(json.dumps(pretes, ensure_ascii=False, indent=1), encoding="utf-8")
            self.stdout.write(f"\nQuestions prêtes pour {', '.join(options['methodes'])} : {len(pretes)} "
                              f"(liste : {moteur.FICHIER_QUESTIONS.name})")
        except OSError as erreur:
            self.stdout.write(self.style.WARNING(f"Liste des questions non écrite : {erreur}"))

        self.stdout.write("")
        if tout_va_bien:
            self.stdout.write(self.style.SUCCESS(
                f"Tout est prêt. Cache : {moteur.FICHIER_CACHE.name} ({len(moteur.lire_cache())} vecteurs). "
                "Lancez : python manage.py runserver"))
        else:
            self.stdout.write(self.style.WARNING(
                "Au moins une vérification a échoué : l'application reste utilisable en TF-IDF (voir le message ci-dessus)."))
