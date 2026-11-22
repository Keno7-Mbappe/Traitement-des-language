"""À lancer UNE FOIS, avec internet, avant la démonstration. Au choix, un des deux services :

  - le serveur Colab du projet (notebook Serveur_Embeddings_Colab.ipynb), qui affiche les deux lignes à copier :
        $env:PROJET8_SERVEUR = "https://....trycloudflare.com"   (PowerShell)     set PROJET8_SERVEUR=...   (cmd)
        $env:PROJET8_CLE = "la clé affichée par Colab"
  - le service d'inférence de Hugging Face :
        $env:HF_TOKEN = "votre_jeton"   (PowerShell)     set HF_TOKEN=votre_jeton   (cmd)
  puis :
        python manage.py preparer_en_ligne

La commande :
  1. vérifie que le service renvoie bien les MÊMES vecteurs que ceux du notebook (on calcule en ligne le vecteur
     de quelques passages et on le compare à celui enregistré dans le dossier artefacts/) ;
  2. calcule les vecteurs des questions d'exemple et les garde dans un cache : ces questions marcheront ensuite même
     si la connexion tombe pendant la démonstration.
Le jeton et la clé ne sont jamais affichés ni enregistrés. Seul le texte des passages et des questions est envoyé au service."""
import numpy as np
from django.core.management.base import BaseCommand, CommandError

from recherche import moteur, views

SEUIL_COSINUS = 0.99          # en dessous, le service ne renvoie pas le même vecteur que le notebook
PASSAGES_TESTES = (0, 57, 150)


class Command(BaseCommand):
    help = "Vérifie le service d'embeddings en ligne (serveur Colab ou Hugging Face) et prépare le cache des questions d'exemple."

    def add_arguments(self, parser):
        parser.add_argument("--methodes", nargs="+", default=list(moteur.MODELES), choices=list(moteur.MODELES),
                            help="méthodes à embeddings à vérifier (par défaut : toutes)")

    def handle(self, *args, **options):
        if not (moteur.adresse_serveur() or moteur.jeton_hf()):
            raise CommandError("Aucun service en ligne configuré. Définissez PROJET8_SERVEUR et PROJET8_CLE (serveur Colab : "
                               "lancez Serveur_Embeddings_Colab.ipynb, il affiche les commandes) ou HF_TOKEN (jeton gratuit "
                               "sur huggingface.co > Settings > Access Tokens), puis relancez la commande.")
        service = {"serveur": "serveur Colab", "huggingface": "Hugging Face", None: "aucun (PROJET8_MODE ?)"}[moteur.service_en_ligne()]
        self.stdout.write(f"Service utilisé : {service}")
        passages = moteur.charger_passages()
        exemples = [moteur.nettoyer(q) for q in views.EXEMPLES_QUI_MARCHENT + views.EXEMPLES_DIFFICILES]
        tout_va_bien = True

        for methode in options["methodes"]:
            nom_modele, _, prefixe_passage = moteur.MODELES[methode]
            self.stdout.write(f"\n=== {moteur.METHODES[methode]} ({nom_modele}) ===")
            matrice = moteur.matrice_dense(methode)

            self.stdout.write("1) vecteurs des passages : service comparé au notebook")
            for i in PASSAGES_TESTES:
                try:
                    v = moteur.vecteur_service(methode, prefixe_passage + passages.iloc[i]["texte"])
                except moteur.MethodeIndisponible as raison:
                    self.stdout.write(self.style.ERROR(f"   ÉCHEC : {raison}"))
                    tout_va_bien = False
                    break
                cosinus = float(np.dot(v, matrice[i]) / np.linalg.norm(matrice[i]))
                etat = self.style.SUCCESS("OK") if cosinus >= SEUIL_COSINUS else self.style.ERROR("ÉCART")
                self.stdout.write(f"   passage {passages.iloc[i]['id']} : cosinus = {cosinus:.4f}  {etat}")
                tout_va_bien &= cosinus >= SEUIL_COSINUS
            else:
                self.stdout.write("2) vecteurs des questions d'exemple : calcul et mise en cache")
                for question in exemples:
                    try:
                        moteur.vecteur_requete.cache_clear()
                        moteur.vecteur_requete(methode, question)
                        self.stdout.write(f"   {question[:70]}  {self.style.SUCCESS('OK')}")
                    except moteur.MethodeIndisponible as raison:
                        self.stdout.write(self.style.ERROR(f"   {question[:60]} : {raison}"))
                        tout_va_bien = False

        self.stdout.write("")
        if tout_va_bien:
            self.stdout.write(self.style.SUCCESS(
                f"Tout est prêt. Cache : {moteur.FICHIER_CACHE.name} ({len(moteur.lire_cache())} questions). "
                "Lancez : python manage.py runserver"))
        else:
            self.stdout.write(self.style.WARNING(
                "Au moins une vérification a échoué : l'application reste utilisable en TF-IDF (voir le message ci-dessus)."))
