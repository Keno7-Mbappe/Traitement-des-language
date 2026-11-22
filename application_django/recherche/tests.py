"""Tests de l'application : python manage.py test

Aucune base de données, aucun jeton, aucun modèle et aucune connexion à Internet ne sont nécessaires : le service
Hugging Face est remplacé par un faux client, et le serveur Colab par un petit serveur HTTP local (ou par le vrai code du
serveur, lu dans Serveur_Embeddings_Colab.ipynb, avec de faux modèles). Chaque test part d'un environnement vierge
(pas de jeton, pas de serveur, pas de modèle local, cache vide)."""
import json
import os
import re
import socket
import tempfile
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from decimal import ROUND_HALF_UP, Decimal
from io import StringIO
from pathlib import Path
from unittest import mock, skipUnless
from urllib.parse import quote

import numpy as np
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase
from django.urls import reverse

from . import donnees_resultats, moteur, views

DIMENSIONS = {"intfloat/multilingual-e5-small": 384, "sentence-transformers/LaBSE": 768}


def faits_du_passage(id_passage):
    return set(moteur.charger_passages().set_index("id").loc[id_passage, "faits"].split(";"))


class FauxClient:
    """Remplace InferenceClient : renvoie des vecteurs de la bonne dimension, ou une erreur. Garde la trace des appels."""

    def __init__(self, erreurs=None, sortie=None, fidele=False):
        self.appels = []
        self.erreurs = list(erreurs or [])      # erreurs à lever, une par appel, avant de répondre normalement
        self.sortie = sortie                    # sortie imposée (pour tester les formats inattendus)
        self.fidele = fidele                    # vrai : renvoie les vrais vecteurs du notebook pour les passages testés

    def feature_extraction(self, texte, model):
        self.appels.append((texte, model))
        if self.erreurs:
            raise self.erreurs.pop(0)
        if self.sortie is not None:
            return self.sortie
        if self.fidele:
            passages = moteur.charger_passages()
            methode = next(m for m, (nom, _, _) in moteur.MODELES.items() if moteur.url_service(nom) == model)
            prefixe = moteur.MODELES[methode][2]
            for i in range(len(passages)):
                if prefixe + passages.iloc[i]["texte"] == texte:
                    return moteur.matrice_dense(methode)[i]
        # vecteur proche de celui d'un passage tiré selon le texte (comme un vrai modèle : des similarités surtout positives)
        methode = next(m for m, (nom, _, _) in moteur.MODELES.items() if moteur.url_service(nom) == model)
        matrice = moteur.matrice_dense(methode)
        graine = abs(hash(texte))
        bruit = np.random.default_rng(graine % (2 ** 32)).normal(size=matrice.shape[1]) * 0.02
        return matrice[graine % len(matrice)] + bruit


class IsoleTestCase(SimpleTestCase):
    def setUp(self):
        environnement = mock.patch.dict(os.environ)
        environnement.start()
        self.addCleanup(environnement.stop)
        for variable in ("HF_TOKEN", "PROJET8_MODE", "PROJET8_SERVEUR", "PROJET8_CLE"):
            os.environ.pop(variable, None)
        dossier = tempfile.TemporaryDirectory()
        self.addCleanup(dossier.cleanup)
        for patch in (mock.patch.object(moteur, "FICHIER_CACHE", Path(dossier.name) / "cache.json"),
                      mock.patch.object(moteur, "FICHIER_QUESTIONS", Path(dossier.name) / "questions_pretes.json"),
                      mock.patch.object(moteur, "modele_local_pret", return_value=False)):
            patch.start()
            self.addCleanup(patch.stop)
        self.reinitialiser_caches()
        self.addCleanup(self.reinitialiser_caches)

    @staticmethod
    def reinitialiser_caches():
        moteur._cache["charge"] = None
        moteur.vecteur_requete.cache_clear()

    def avec_jeton_et_faux_client(self, faux):
        os.environ["HF_TOKEN"] = "jeton-de-test-0123456789"
        patch = mock.patch.object(moteur, "client_hf", return_value=faux)
        patch.start()
        self.addCleanup(patch.stop)
        return faux


class MoteurTests(IsoleTestCase):
    def test_artefacts_coherents(self):
        passages = moteur.charger_passages()
        self.assertEqual(len(passages), 232)
        _, matrice, _ = moteur.charger_tfidf()
        self.assertEqual(matrice.shape[0], len(passages))
        self.assertEqual(moteur.matrice_dense("embeddings").shape, (232, 384))
        self.assertEqual(moteur.matrice_dense("labse").shape, (232, 768))

    def test_reproduit_le_notebook(self):
        # Dans le notebook (exécution Colab), « bibliotheque horaires » renvoie d'abord la carte d'étudiant.
        self.assertEqual(moteur.rechercher("bibliotheque horaires")[0]["id"], "guide_inscription_fr-14")

    def test_question_en_francais(self):
        top3 = [r["id"] for r in moteur.rechercher("Combien coûtent les frais de scolarité en Master ?")]
        self.assertIn("montant_frais", faits_du_passage(top3[0]))

    def test_question_en_anglais(self):
        top3 = [r["id"] for r in moteur.rechercher("Do I need a visa to study here?")]
        self.assertIn("etudiants_etrangers", faits_du_passage(top3[0]))

    def test_question_en_somali(self):
        top3 = [r["id"] for r in moteur.rechercher("Maktabaddu goorma ayay furan tahay?")]
        self.assertIn("horaires_bibliotheque", faits_du_passage(top3[0]))

    def test_cas_difficile_connu(self):
        # Limite documentée dans le notebook : sans mot commun, TF-IDF ne trouve pas le passage sur l'étalement des paiements.
        top3 = [r["id"] for r in moteur.rechercher("Je n'ai plus d'argent pour payer ma scolarité, que faire ?")]
        self.assertFalse(any("etalement" in faits_du_passage(i) for i in top3))

    def test_trois_resultats_au_plus_et_scores_decroissants(self):
        resultats = moteur.rechercher("frais de scolarité")
        self.assertLessEqual(len(resultats), 3)
        scores = [r["score"] for r in resultats]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([r["rang"] for r in resultats], list(range(1, len(resultats) + 1)))

    def test_aucun_resultat_plutot_qu_un_classement_au_hasard(self):
        self.assertEqual(moteur.rechercher("§§§§ ¤¤¤¤"), [])

    def test_methode_sans_jeton_ni_modele_est_signalee(self):
        ok, raison = moteur.disponibilite("embeddings")
        self.assertFalse(ok)
        self.assertIn("HF_TOKEN", raison)
        with self.assertRaises(moteur.MethodeIndisponible):
            moteur.rechercher("frais de scolarité", "embeddings")


class EnLigneTests(IsoleTestCase):
    def test_jeton_rend_les_methodes_disponibles(self):
        self.avec_jeton_et_faux_client(FauxClient())
        self.assertEqual(moteur.disponibilite("embeddings"), (True, "en ligne"))
        self.assertEqual(moteur.disponibilite("melange_labse"), (True, "en ligne"))
        self.assertTrue(moteur.mode_en_ligne_actif())

    def test_mode_local_force_ignore_le_jeton(self):
        self.avec_jeton_et_faux_client(FauxClient())
        os.environ["PROJET8_MODE"] = "local"
        self.assertIsNone(moteur.mode_embeddings("embeddings"))

    def test_vecteur_e5_prefixe_et_normalise(self):
        faux = self.avec_jeton_et_faux_client(FauxClient())
        v = moteur.vecteur_requete("embeddings", "frais de scolarité")
        self.assertEqual(v.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
        self.assertEqual(faux.appels[0], ("query: frais de scolarité", moteur.url_service("intfloat/multilingual-e5-small")))

    def test_labse_sans_prefixe(self):
        faux = self.avec_jeton_et_faux_client(FauxClient())
        moteur.vecteur_requete("labse", "frais de scolarité")
        self.assertEqual(faux.appels[0], ("frais de scolarité", moteur.url_service("sentence-transformers/LaBSE")))

    def test_recherche_avec_embeddings_et_melange(self):
        self.avec_jeton_et_faux_client(FauxClient())
        for methode in ("embeddings", "labse", "melange", "melange_labse"):
            resultats = moteur.rechercher("frais de scolarité", methode)
            self.assertEqual(len(resultats), 3, methode)
            scores = [r["score"] for r in resultats]
            self.assertEqual(scores, sorted(scores, reverse=True), methode)

    def test_cache_permet_de_rejouer_sans_connexion(self):
        faux = self.avec_jeton_et_faux_client(FauxClient())
        moteur.rechercher("frais de scolarité", "embeddings")
        self.assertEqual(len(faux.appels), 1)
        self.assertTrue(moteur.FICHIER_CACHE.exists())
        # plus de jeton, caches mémoire vidés : seule la question déjà calculée doit encore marcher
        os.environ.pop("HF_TOKEN")
        self.reinitialiser_caches()
        self.assertEqual(moteur.disponibilite("embeddings"), (True, "questions déjà calculées seulement"))
        self.assertEqual(len(moteur.rechercher("frais de scolarité", "embeddings")), 3)
        self.assertEqual(len(faux.appels), 1, "aucun nouvel appel en ligne")
        with self.assertRaises(moteur.MethodeIndisponible):
            moteur.rechercher("une autre question jamais posée", "embeddings")

    def test_jeton_refuse(self):
        self.avec_jeton_et_faux_client(FauxClient(erreurs=[Exception("401 Client Error: Unauthorized for url")]))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "jeton"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_credits_epuises(self):
        self.avec_jeton_et_faux_client(FauxClient(erreurs=[Exception("402 Client Error: Payment Required: you have exceeded your monthly included credits")]))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "crédits"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_modele_non_servi(self):
        self.avec_jeton_et_faux_client(FauxClient(erreurs=[Exception("404 Client Error: Not Found")]))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "pas servi"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_adresse_du_service_complete(self):
        self.assertEqual(moteur.url_service("intfloat/multilingual-e5-small"),
                         "https://router.huggingface.co/hf-inference/models/intfloat/multilingual-e5-small/pipeline/feature-extraction")

    def test_pas_de_connexion(self):
        self.avec_jeton_et_faux_client(FauxClient(erreurs=[ConnectionError("Connection aborted")]))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "connexion"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_le_jeton_n_apparait_jamais_dans_les_messages(self):
        faux = self.avec_jeton_et_faux_client(FauxClient(erreurs=[Exception("échec avec jeton-de-test-0123456789 dans l'adresse")]))
        with self.assertRaises(moteur.MethodeIndisponible) as contexte:
            moteur.vecteur_requete("embeddings", "frais")
        self.assertNotIn("jeton-de-test-0123456789", str(contexte.exception))

    def test_modele_en_cours_de_chargement_puis_succes(self):
        faux = self.avec_jeton_et_faux_client(FauxClient(erreurs=[Exception("503 Service Unavailable: model is loading")]))
        with mock.patch.object(moteur.time, "sleep"):
            v = moteur.vecteur_requete("embeddings", "frais")
        self.assertEqual(v.shape, (384,))
        self.assertEqual(len(faux.appels), 2)

    def test_mauvaise_dimension_refusee(self):
        self.avec_jeton_et_faux_client(FauxClient(sortie=np.ones(10)))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "dimension inattendue"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_vecteur_par_mot_refuse(self):
        self.avec_jeton_et_faux_client(FauxClient(sortie=np.ones((5, 384))))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "par mot"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_matrice_a_une_ligne_acceptee(self):
        self.avec_jeton_et_faux_client(FauxClient(sortie=np.ones((1, 384))))
        self.assertEqual(moteur.vecteur_requete("embeddings", "frais").shape, (384,))


class CommandeTests(IsoleTestCase):
    def test_refuse_sans_jeton(self):
        with self.assertRaises(CommandError):
            call_command("preparer_en_ligne", stdout=StringIO())

    def test_service_fidele_au_notebook(self):
        self.avec_jeton_et_faux_client(FauxClient(fidele=True))
        sortie = StringIO()
        call_command("preparer_en_ligne", stdout=sortie)
        texte = sortie.getvalue()
        self.assertIn("cosinus = 1.0000", texte)
        self.assertIn("Tout est prêt", texte)
        self.assertGreaterEqual(len(moteur.lire_cache()), 12)       # 6 questions d'exemple x 2 modèles

    def test_service_infidele_est_signale(self):
        self.avec_jeton_et_faux_client(FauxClient())                 # vecteurs au hasard : différents de ceux du notebook
        sortie = StringIO()
        call_command("preparer_en_ligne", stdout=sortie)
        self.assertIn("ÉCART", sortie.getvalue())
        self.assertIn("Au moins une vérification a échoué", sortie.getvalue())


class VuesTests(IsoleTestCase):
    def test_page_d_accueil(self):
        reponse = self.client.get(reverse("accueil"))
        self.assertEqual(reponse.status_code, 200)
        self.assertContains(reponse, "Recherche dans les documents universitaires")
        self.assertNotContains(reponse, "envoyé à Hugging Face")

    def test_page_avec_resultats(self):
        reponse = self.client.get(reverse("accueil"), {"q": "Combien coûtent les frais de scolarité en Master ?"})
        self.assertEqual(reponse.status_code, 200)
        self.assertContains(reponse, "Montant des droits")
        self.assertContains(reponse, "Français")

    def test_resultat_somali_affiche_la_langue(self):
        reponse = self.client.get(reverse("accueil"), {"q": "Maktabaddu goorma ayay furan tahay?"})
        self.assertContains(reponse, "Soomaali")

    def test_methode_indisponible_retombe_sur_tfidf(self):
        reponse = self.client.get(reverse("accueil"), {"q": "frais de scolarité", "methode": "melange"})
        self.assertEqual(reponse.status_code, 200)
        self.assertContains(reponse, "pas disponible sur cet ordinateur")
        self.assertContains(reponse, "TF-IDF")

    def test_page_en_ligne_avec_melange(self):
        self.avec_jeton_et_faux_client(FauxClient())
        reponse = self.client.get(reverse("accueil"), {"q": "frais de scolarité", "methode": "melange"})
        self.assertContains(reponse, "Mélange TF-IDF + e5-small")
        self.assertNotContains(reponse, "pas disponible sur cet ordinateur")
        self.assertContains(reponse, "envoyé à Hugging Face")

    def test_page_en_ligne_en_panne_retombe_sur_tfidf(self):
        self.avec_jeton_et_faux_client(FauxClient(erreurs=[ConnectionError("Connection aborted")]))
        reponse = self.client.get(reverse("accueil"), {"q": "frais de scolarité", "methode": "melange"})
        self.assertEqual(reponse.status_code, 200)
        self.assertContains(reponse, "pas de connexion internet")
        self.assertContains(reponse, "Les résultats ci-dessous viennent de la méthode TF-IDF")

    def test_requete_trop_longue(self):
        reponse = self.client.get(reverse("accueil"), {"q": "a" * 400})
        self.assertContains(reponse, "trop longue")

    def test_api_json(self):
        reponse = self.client.get(reverse("api_recherche"), {"q": "bibliotheque horaires"})
        self.assertEqual(reponse.status_code, 200)
        donnees = reponse.json()
        self.assertEqual(set(donnees), {"requete", "methode", "resultats"})
        self.assertEqual(donnees["methode"], "tfidf")
        self.assertEqual(set(donnees["resultats"][0]), {"rang", "score", "id", "source", "langue", "titre", "passage"})

    def test_api_en_ligne(self):
        self.avec_jeton_et_faux_client(FauxClient())
        reponse = self.client.get(reverse("api_recherche"), {"q": "frais de scolarité", "methode": "labse"})
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(reponse.json()["methode"], "labse")

    def test_api_sans_requete(self):
        self.assertEqual(self.client.get(reverse("api_recherche")).status_code, 400)

    def test_api_methode_inconnue(self):
        self.assertEqual(self.client.get(reverse("api_recherche"), {"q": "x", "methode": "inconnue"}).status_code, 400)

    def test_api_methode_indisponible(self):
        reponse = self.client.get(reverse("api_recherche"), {"q": "frais", "methode": "labse"})
        self.assertEqual(reponse.status_code, 503)

    def test_api_service_en_ligne_refuse_le_jeton(self):
        self.avec_jeton_et_faux_client(FauxClient(erreurs=[Exception("401 Client Error: Unauthorized")]))
        reponse = self.client.get(reverse("api_recherche"), {"q": "frais", "methode": "embeddings"})
        self.assertEqual(reponse.status_code, 503)
        self.assertIn("jeton", reponse.json()["erreur"])


# ----------------------------------------------------------------------------------------------- serveur Colab (plan B)
def methode_du_modele(nom_modele):
    return next(m for m, (nom, _, _) in moteur.MODELES.items() if nom == nom_modele)


class FauxServeur:
    """Petit serveur HTTP local qui joue le rôle du serveur Colab. reponse(corps JSON) -> (code HTTP, objet JSON)."""

    def __init__(self, reponse):
        self.requetes = []                       # (adresse, en-têtes, corps JSON) de chaque appel reçu
        self.arrete = False
        faux = self

        class Gestionnaire(BaseHTTPRequestHandler):
            def do_POST(self):
                corps = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                faux.requetes.append((self.path, self.headers, corps))
                code, objet = reponse(corps)
                donnees = json.dumps(objet).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Length", str(len(donnees)))
                self.end_headers()
                self.wfile.write(donnees)

            def log_message(self, *arguments):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Gestionnaire)
        self.adresse = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def arreter(self):
        if not self.arrete:
            self.arrete = True
            self.httpd.shutdown()
            self.httpd.server_close()


def adresse_sans_serveur():
    """Adresse locale où personne n'écoute (serveur Colab arrêté)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class ServeurColabTests(IsoleTestCase):
    def avec_faux_serveur(self, reponse=None, cle="cle-de-test"):
        if reponse is None:
            def reponse(corps):
                return 200, {"vecteurs": [moteur.matrice_dense(methode_du_modele(corps["modele"]))[0].tolist()]}
        faux = FauxServeur(reponse)
        self.addCleanup(faux.arreter)
        os.environ["PROJET8_SERVEUR"] = faux.adresse + "/"          # le « / » final doit être ignoré
        os.environ["PROJET8_CLE"] = cle
        return faux

    def test_les_methodes_a_embeddings_deviennent_disponibles(self):
        self.avec_faux_serveur()
        self.assertEqual(moteur.disponibilite("embeddings"), (True, "serveur Colab"))
        self.assertEqual(moteur.disponibilite("melange_labse"), (True, "serveur Colab"))
        self.assertEqual(moteur.service_en_ligne(), "serveur")
        self.assertTrue(moteur.mode_en_ligne_actif())

    def test_requete_envoyee_au_serveur_avec_prefixe_cle_et_agent(self):
        faux = self.avec_faux_serveur()
        v = moteur.vecteur_requete("embeddings", "frais de scolarité")
        self.assertEqual(v.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
        adresse, en_tetes, corps = faux.requetes[0]
        self.assertEqual(adresse, "/embeddings")
        self.assertEqual(corps, {"modele": "intfloat/multilingual-e5-small", "textes": ["query: frais de scolarité"]})
        self.assertEqual(en_tetes.get("Authorization"), "Bearer cle-de-test")
        self.assertEqual(en_tetes.get("User-Agent"), moteur.AGENT_HTTP)
        self.assertEqual(en_tetes.get("Content-Type"), "application/json")

    def test_labse_sans_prefixe(self):
        faux = self.avec_faux_serveur()
        self.assertEqual(moteur.vecteur_requete("labse", "frais de scolarité").shape, (768,))
        self.assertEqual(faux.requetes[0][2], {"modele": "sentence-transformers/LaBSE", "textes": ["frais de scolarité"]})

    def test_recherche_avec_les_quatre_methodes(self):
        self.avec_faux_serveur()
        for methode in ("embeddings", "labse", "melange", "melange_labse"):
            self.assertEqual(len(moteur.rechercher("frais de scolarité", methode)), 3, methode)

    def test_le_serveur_passe_avant_hugging_face(self):
        self.avec_faux_serveur()
        os.environ["HF_TOKEN"] = "jeton-de-test-0123456789"
        self.assertEqual(moteur.mode_embeddings("embeddings"), "serveur")
        os.environ["PROJET8_MODE"] = "en_ligne"                     # on peut forcer Hugging Face
        self.assertEqual(moteur.mode_embeddings("embeddings"), "en_ligne")
        os.environ["PROJET8_MODE"] = "local"
        self.assertIsNone(moteur.mode_embeddings("embeddings"))

    def test_cle_refusee(self):
        self.avec_faux_serveur(lambda corps: (401, {"erreur": "clé absente ou incorrecte"}))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "clé du serveur Colab refusée"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_serveur_arrete(self):
        os.environ["PROJET8_SERVEUR"] = adresse_sans_serveur()
        os.environ["PROJET8_CLE"] = "cle-de-test"
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "injoignable"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_tunnel_ferme_cote_cloudflare(self):
        self.avec_faux_serveur(lambda corps: (530, {}))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "relancer"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_la_cle_n_apparait_jamais_dans_les_messages(self):
        self.avec_faux_serveur(lambda corps: (400, {"erreur": "recu cle-de-test dans la demande"}))
        with self.assertRaises(moteur.MethodeIndisponible) as contexte:
            moteur.vecteur_requete("embeddings", "frais")
        self.assertNotIn("cle-de-test", str(contexte.exception))
        self.assertIn("***", str(contexte.exception))

    def test_mauvaise_dimension_refusee(self):
        self.avec_faux_serveur(lambda corps: (200, {"vecteurs": [[1.0] * 10]}))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "dimension inattendue"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_reponse_qui_n_est_pas_celle_du_serveur_du_projet(self):
        self.avec_faux_serveur(lambda corps: (200, {"autre": 1}))
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "réponse illisible"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_adresse_incomplete(self):
        os.environ["PROJET8_SERVEUR"] = "exemple.trycloudflare.com"
        os.environ["PROJET8_CLE"] = "cle-de-test"
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "adresse complète"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_cache_permet_de_rejouer_apres_l_arret_du_serveur(self):
        faux = self.avec_faux_serveur()
        moteur.rechercher("frais de scolarité", "embeddings")
        faux.arreter()
        for variable in ("PROJET8_SERVEUR", "PROJET8_CLE"):
            os.environ.pop(variable)
        self.reinitialiser_caches()
        self.assertEqual(moteur.disponibilite("embeddings"), (True, "questions déjà calculées seulement"))
        self.assertEqual(len(moteur.rechercher("frais de scolarité", "embeddings")), 3)
        self.assertEqual(len(faux.requetes), 1, "aucun nouvel appel au serveur")

    def test_page_indique_que_la_question_va_au_serveur_colab(self):
        self.avec_faux_serveur()
        reponse = self.client.get(reverse("accueil"), {"q": "frais de scolarité", "methode": "melange"})
        self.assertContains(reponse, "serveur d'embeddings du projet (Colab)")
        self.assertNotContains(reponse, "envoyé à Hugging Face")
        self.assertNotContains(reponse, "pas disponible sur cet ordinateur")

    def test_page_serveur_en_panne_retombe_sur_tfidf(self):
        os.environ["PROJET8_SERVEUR"] = adresse_sans_serveur()
        os.environ["PROJET8_CLE"] = "cle-de-test"
        reponse = self.client.get(reverse("accueil"), {"q": "frais de scolarité", "methode": "melange"})
        self.assertEqual(reponse.status_code, 200)
        self.assertContains(reponse, "injoignable")
        self.assertContains(reponse, "Les résultats ci-dessous viennent de la méthode TF-IDF")

    def test_api_cle_refusee(self):
        self.avec_faux_serveur(lambda corps: (401, {"erreur": "clé absente ou incorrecte"}))
        reponse = self.client.get(reverse("api_recherche"), {"q": "frais", "methode": "embeddings"})
        self.assertEqual(reponse.status_code, 503)
        self.assertIn("clé", reponse.json()["erreur"])

    def test_commande_sans_aucun_service(self):
        with self.assertRaisesMessage(CommandError, "PROJET8_SERVEUR"):
            call_command("preparer_en_ligne", stdout=StringIO())


# Le vrai code du serveur Colab (cellule du notebook), avec de faux modèles : on vérifie que l'application et le serveur parlent
# bien le même langage (adresses, en-têtes, forme du JSON, clé, limites), sans Colab ni Internet.
RACINE_PROJET = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_SERVEUR = RACINE_PROJET / "Serveur_Embeddings_Colab.ipynb"


def code_du_serveur_colab():
    """Exécute la cellule du notebook qui DÉFINIT le serveur (sans rien démarrer) et retourne son espace de noms."""
    notebook = json.loads(NOTEBOOK_SERVEUR.read_text(encoding="utf-8"))
    for cellule in notebook["cells"]:
        source = "".join(cellule["source"])
        if cellule["cell_type"] == "code" and "def creer_serveur" in source:
            espace = {"print": lambda *arguments, **options: None}          # pas de journal pendant les tests
            exec(compile(source, "Serveur_Embeddings_Colab.ipynb (cellule du serveur)", "exec"), espace)
            return espace
    raise AssertionError("cellule du serveur introuvable dans le notebook")


class FauxModele:
    """Remplace SentenceTransformer : vrai vecteur du notebook pour un passage connu, vecteur stable sinon."""

    def __init__(self, methode):
        self.methode = methode

    def encode(self, textes, normalize_embeddings=True, show_progress_bar=False):
        matrice = moteur.matrice_dense(self.methode)
        prefixe = moteur.MODELES[self.methode][2]
        connus = {prefixe + t: i for i, t in enumerate(moteur.charger_passages()["texte"])}
        sortie = []
        for texte in textes:
            if texte in connus:
                sortie.append(matrice[connus[texte]])
            else:
                v = np.random.default_rng(abs(hash(texte)) % (2 ** 32)).normal(size=matrice.shape[1])
                sortie.append(v / np.linalg.norm(v))
        return np.array(sortie)


@skipUnless(NOTEBOOK_SERVEUR.exists(), "notebook du serveur Colab absent")
class ServeurDuNotebookTests(IsoleTestCase):
    def setUp(self):
        super().setUp()
        self.espace = code_du_serveur_colab()
        modeles = {nom: FauxModele(methode) for methode, (nom, _, _) in moteur.MODELES.items()}
        self.serveur = self.espace["creer_serveur"](modeles, "cle-du-notebook", port=0)
        threading.Thread(target=self.serveur.serve_forever, daemon=True).start()
        self.addCleanup(self.serveur.server_close)
        self.addCleanup(self.serveur.shutdown)                      # les nettoyages s'exécutent dans l'ordre inverse
        self.adresse = f"http://127.0.0.1:{self.serveur.server_address[1]}"
        os.environ["PROJET8_SERVEUR"] = self.adresse
        os.environ["PROJET8_CLE"] = "cle-du-notebook"

    def appeler(self, chemin, cle="cle-du-notebook", donnees=None):
        return self.espace["appeler"](self.adresse, chemin, cle, donnees)

    def test_etat_du_serveur_sans_cle(self):
        self.assertEqual(self.espace["appeler"](self.adresse, "/sante"),
                         {"etat": "ok", "modeles": {"intfloat/multilingual-e5-small": 384, "sentence-transformers/LaBSE": 768}})

    def test_l_application_obtient_des_vecteurs_valides(self):
        for methode, dimension in (("embeddings", 384), ("labse", 768)):
            v = moteur.vecteur_requete(methode, "frais de scolarité")
            self.assertEqual(v.shape, (dimension,))
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_recherche_avec_les_quatre_methodes(self):
        for methode in ("embeddings", "labse", "melange", "melange_labse"):
            self.assertEqual(len(moteur.rechercher("frais de scolarité", methode)), 3, methode)

    def test_commande_preparer_en_ligne_avec_le_serveur(self):
        sortie = StringIO()
        call_command("preparer_en_ligne", stdout=sortie)
        texte = sortie.getvalue()
        self.assertIn("Service utilisé : serveur Colab", texte)
        self.assertIn("cosinus = 1.0000", texte)
        self.assertIn("Tout est prêt", texte)
        self.assertGreaterEqual(len(moteur.lire_cache()), 12)       # 6 questions d'exemple x 2 modèles

    def test_mauvaise_cle(self):
        os.environ["PROJET8_CLE"] = "autre-cle"
        with self.assertRaisesMessage(moteur.MethodeIndisponible, "clé du serveur Colab refusée"):
            moteur.vecteur_requete("embeddings", "frais")

    def test_appels_refuses_par_le_serveur(self):
        e5 = "intfloat/multilingual-e5-small"
        cas = {
            "sans clé": (None, {"modele": e5, "textes": ["a"]}, 401),
            "modèle inconnu": ("cle-du-notebook", {"modele": "autre/modele", "textes": ["a"]}, 400),
            "aucun texte": ("cle-du-notebook", {"modele": e5, "textes": []}, 400),
            "trop de textes": ("cle-du-notebook", {"modele": e5, "textes": ["a"] * 33}, 400),
            "texte trop long": ("cle-du-notebook", {"modele": e5, "textes": ["a" * 2001]}, 400),
            "pas du texte": ("cle-du-notebook", {"modele": e5, "textes": [3]}, 400),
            "champ manquant": ("cle-du-notebook", {"modele": e5}, 400),
        }
        for nom, (cle, donnees, code) in cas.items():
            with self.subTest(nom), self.assertRaises(urllib.error.HTTPError) as contexte:
                self.appeler("/embeddings", cle, donnees)
            self.assertEqual(contexte.exception.code, code)

    def test_adresse_inconnue(self):
        with self.assertRaises(urllib.error.HTTPError) as contexte:
            self.appeler("/autre", None)
        self.assertEqual(contexte.exception.code, 404)

    def test_modele_non_autorise_refuse_au_demarrage(self):
        with self.assertRaises(ValueError):
            self.espace["creer_serveur"]({"autre/modele": object()}, "cle", port=0)


# ----------------------------------------------------------------------------------------------- refonte de l'interface
class PagesTests(IsoleTestCase):
    def test_pages_internes(self):
        for nom, texte in (("methode", "la chaîne de traitement"), ("resultats", "Ce que les mesures"), ("corpus", "Quatorze documents")):
            reponse = self.client.get(reverse(nom))
            self.assertEqual(reponse.status_code, 200, nom)
            self.assertContains(reponse, texte)

    def test_navigation_commune_et_page_courante(self):
        for nom in ("accueil", "methode", "resultats", "corpus"):
            html = self.client.get(reverse(nom)).content.decode()
            for cible in ("accueil", "methode", "resultats", "corpus"):
                self.assertIn(f'href="{reverse(cible)}"', html, f"{nom} -> {cible}")
            self.assertEqual(html.count('aria-current="page"'), 1, nom)
            self.assertIn('class="lien-evitement"', html, "lien d'évitement pour le clavier")

    def test_page_introuvable_au_style_du_site(self):
        reponse = self.client.get("/n-existe-pas/")
        self.assertContains(reponse, "Scriptorium", status_code=404)
        self.assertContains(reponse, "Retour à la recherche", status_code=404)

    def test_fichiers_statiques_references_existent(self):
        dossier = Path(__file__).resolve().parent / "static"
        html = self.client.get(reverse("accueil")).content.decode()
        adresses = set(re.findall(r'(?:href|src)="/static/([^"]+)"', html))
        self.assertTrue({"recherche/css/app.css", "recherche/js/app.js", "recherche/js/carte.js", "recherche/favicon.svg"} <= adresses, adresses)
        for adresse in adresses:
            self.assertTrue((dossier / adresse).exists(), adresse)
        css = (dossier / "recherche" / "css" / "app.css").read_text(encoding="utf-8")
        for police in re.findall(r'url\("\.\./fonts/([^"]+)"\)', css):
            self.assertTrue((dossier / "recherche" / "fonts" / police).exists(), police)
        self.assertEqual(css.count("{"), css.count("}"), "accolades équilibrées dans la feuille de style")

    def test_aucune_virgule_decimale_dans_les_styles_et_les_donnees(self):
        # Avec LANGUAGE_CODE = « fr », un nombre non protégé s'écrit « 0,84 » et casse le CSS et le JavaScript.
        pages = [self.client.get(reverse(n)).content.decode() for n in ("accueil", "methode", "resultats", "corpus")]
        pages.append(self.client.get(reverse("accueil"), {"q": "frais de scolarité"}).content.decode())
        for html in pages:
            self.assertIsNone(re.search(r"--[a-z-]+:\s*-?\d+,\d", html))
            self.assertIsNone(re.search(r'data-[a-z-]+="-?\d+,\d', html))

    def test_methode_affiche_une_vraie_reponse_de_l_api(self):
        reponse = self.client.get(reverse("methode"))
        self.assertContains(reponse, "guide_inscription_fr-14")
        self.assertContains(reponse, "Lire Precision@3, Hit@3 et MRR")

    def test_accueil_sans_requete_affiche_les_chiffres_du_projet(self):
        html = self.client.get(reverse("accueil")).content.decode()
        for attendu in ('data-compter="232"', 'data-compter="14"', 'data-compter="12500"', 'data-compter="84"', 'data-compter="88"', 'data-compter="57"'):
            self.assertIn(attendu, html)


class RechercheInterfaceTests(IsoleTestCase):
    def donnees_carte(self, reponse):
        trouve = re.search(r'<script id="carte-donnees" type="application/json">(.*?)</script>', reponse.content.decode(), re.S)
        self.assertIsNotNone(trouve)
        return json.loads(trouve.group(1))

    def test_resultats_avec_mots_surlignes_et_carte(self):
        reponse = self.client.get(reverse("accueil"), {"q": "Combien coûtent les frais de scolarité en Master ?"})
        self.assertContains(reponse, "<mark>scolarité</mark>")
        self.assertContains(reponse, "Mots communs")
        carte = self.donnees_carte(reponse)
        self.assertEqual(len(carte["points"]), 232)
        self.assertEqual(len(carte["resultats"]), 3)
        self.assertIsNone(carte["question"], "TF-IDF n'a pas de vecteur e5 : la question n'est pas placée")

    def test_les_mots_outils_ne_sont_pas_surlignes(self):
        reponse = self.client.get(reverse("accueil"), {"q": "les frais de scolarité"})
        self.assertNotContains(reponse, "<mark>les</mark>")
        self.assertNotContains(reponse, "<mark>de</mark>")

    def test_question_placee_sur_la_carte_avec_embeddings(self):
        self.avec_jeton_et_faux_client(FauxClient())
        carte = self.donnees_carte(self.client.get(reverse("accueil"), {"q": "frais de scolarité", "methode": "embeddings"}))
        self.assertEqual(len(carte["question"]), 2)
        carte = self.donnees_carte(self.client.get(reverse("accueil"), {"q": "frais de scolarité", "methode": "labse"}))
        self.assertIsNone(carte["question"], "LaBSE n'est pas dans l'espace e5 : on ne place pas la question")

    def test_sans_requete_la_carte_n_a_pas_de_resultat(self):
        carte = self.donnees_carte(self.client.get(reverse("accueil")))
        self.assertEqual(carte["resultats"], [])

    def test_aucun_resultat_garde_la_carte_et_le_message(self):
        reponse = self.client.get(reverse("accueil"), {"q": "§§§§ ¤¤¤¤"})
        self.assertContains(reponse, "Aucun résultat")

    def test_message_selon_la_methode_quand_aucun_mot_commun(self):
        self.avec_jeton_et_faux_client(FauxClient())
        reponse = self.client.get(reverse("accueil"), {"q": "I cannot afford my tuition at the moment", "methode": "embeddings"})
        self.assertContains(reponse, "appariement par le sens")

    def test_methodes_proposees_sous_forme_de_choix(self):
        html = self.client.get(reverse("accueil")).content.decode()
        self.assertEqual(html.count('name="methode"'), 5)
        self.assertIn('name="methode" value="tfidf" checked', html)
        self.assertEqual(html.count("disabled"), 4, "les quatre méthodes à embeddings sont indisponibles sans service")

    def test_la_requete_est_echappee_dans_la_page(self):
        reponse = self.client.get(reverse("accueil"), {"q": '<script>alert("x")</script>'})
        self.assertNotContains(reponse, '<script>alert("x")</script>')
        self.assertContains(reponse, "&lt;script&gt;")


class SurlignageTests(SimpleTestCase):
    TEXTE = "Les droits de scolarité en Master sont fixes."

    def test_mots_communs_sans_mots_outils(self):
        morceaux, communs = moteur.decouper_surlignage(self.TEXTE, moteur.formes_de_la_requete("Combien coûtent les frais de scolarité en Master ?"), "fr")
        self.assertEqual(communs, ["scolarité", "master"])
        self.assertEqual("".join(t for t, _ in morceaux), self.TEXTE, "le texte est reconstitué à l'identique")
        self.assertEqual([t for t, surligne in morceaux if surligne], ["scolarité", "Master"])

    def test_accents_ignores(self):
        _, communs = moteur.decouper_surlignage(self.TEXTE, moteur.formes_de_la_requete("SCOLARITE"), "fr")
        self.assertEqual(communs, ["scolarité"])

    def test_aucun_mot_commun_entre_langues(self):
        passage = moteur.charger_passages().query("langue == 'so'").iloc[0]["texte"]
        _, communs = moteur.decouper_surlignage(passage, moteur.formes_de_la_requete("Do I need a visa to study here?"), "so")
        self.assertEqual(communs, [])

    def test_formes_frequentes(self):
        frequentes = moteur.formes_frequentes()
        self.assertTrue({"les", "des", "est"} <= frequentes["fr"])
        self.assertTrue({"the", "and"} <= frequentes["en"])
        self.assertIn("iyo", frequentes["so"])
        self.assertNotIn("scolarite", frequentes["fr"])

    def test_html_echappe_et_seulement_mark(self):
        from .views import passage_en_html
        html = str(passage_en_html([('<script>alert(1)</script> & ', False), ("frais", True)]))
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&amp;", html)
        self.assertIn("<mark>frais</mark>", html)
        self.assertNotIn("<script>", html)


class CarteEtCorpusTests(SimpleTestCase):
    def test_carte_des_passages(self):
        carte = moteur.carte_passages()
        self.assertEqual(len(carte["points"]), 232)
        self.assertEqual(carte["variance"], 20, "deux axes : 20 % de la variance, comme dans le rapport")
        self.assertEqual(len({p["id"] for p in carte["points"]}), 232)
        for p in carte["points"]:
            self.assertTrue(-1.0001 <= p["x"] <= 1.0001 and -1.0001 <= p["y"] <= 1.0001)
        centres = {}
        for langue in ("fr", "en", "so"):
            pts = [(p["x"], p["y"]) for p in carte["points"] if p["l"] == langue]
            centres[langue] = np.mean(pts, axis=0)
        for a, b in (("fr", "en"), ("fr", "so"), ("en", "so")):
            self.assertGreater(float(np.linalg.norm(centres[a] - centres[b])), 0.5, f"{a} et {b} forment des îlots séparés")

    def test_position_d_un_passage_sur_la_carte(self):
        carte = moteur.carte_passages()
        x, y = moteur.position_requete(moteur.matrice_dense("embeddings")[5])
        self.assertAlmostEqual(x, carte["points"][5]["x"], places=3)
        self.assertAlmostEqual(y, carte["points"][5]["y"], places=3)
        x, y = moteur.position_requete(np.full(384, 50.0))
        self.assertTrue(abs(x) <= 1.2 and abs(y) <= 1.2, "une question lointaine reste dans le cadre")

    def test_statistiques_du_corpus_comme_dans_le_rapport(self):
        stats = moteur.statistiques_corpus()
        self.assertEqual((len(stats["documents"]), stats["passages"], stats["mots"]), (14, 232, 12500))
        self.assertEqual(stats["langues"]["fr"], {"documents": 7, "passages": 123, "mots": 6943})
        self.assertEqual(stats["langues"]["en"], {"documents": 2, "passages": 42, "mots": 2427})
        self.assertEqual(stats["langues"]["so"], {"documents": 5, "passages": 67, "mots": 3130})
        self.assertEqual(sum(len(d["sections"]) for d in stats["documents"]), 232)


class ResultatsDonneesTests(SimpleTestCase):
    def test_coherence_avec_le_readme(self):
        readme = RACINE_PROJET / "README.md"
        if not readme.exists():
            self.skipTest("README absent")
        noms = {"TF-IDF (baseline)": "tfidf", "Embeddings e5-small": "embeddings", "Embeddings LaBSE": "labse",
                "Mélange TF-IDF + e5-small": "melange", "Mélange TF-IDF + LaBSE": "melange_labse"}
        par_cle = {m["cle"]: m for m in donnees_resultats.METHODES}
        arrondi = lambda v: str(Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP)).replace(".", ",")
        lues = 0
        for ligne in readme.read_text(encoding="utf-8").splitlines():
            cellules = [c.strip().replace("*", "") for c in ligne.strip().strip("|").split("|")]
            if len(cellules) == 5 and cellules[0] in noms:
                m = par_cle[noms[cellules[0]]]
                attendu = [arrondi(m["toutes"]["hit"]), arrondi(m["toutes"]["mrr"]), arrondi(m["meme"]["hit"]), arrondi(m["meme"]["mrr"])]
                self.assertEqual(cellules[1:], attendu, cellules[0])
                lues += 1
        self.assertEqual(lues, 5, "les cinq lignes du tableau du README ont été comparées")

    def test_meilleure_methode(self):
        meilleure = max((m for m in donnees_resultats.METHODES if m["cle"] != "hasard"), key=lambda m: m["toutes"]["hit"])
        self.assertTrue(meilleure.get("meilleur"))
        base = next(m for m in donnees_resultats.METHODES if m["cle"] == "tfidf")
        self.assertEqual(round((meilleure["toutes"]["hit"] - base["toutes"]["hit"]) * 63), 3, "3 requêtes de plus sur 63")

    def test_causes_d_erreur_totalisent_les_89_cas(self):
        causes = donnees_resultats.CAUSES
        totaux = [sum(ligne["valeurs"][j] for ligne in causes["lignes"]) for j in range(3)]
        self.assertEqual(totaux, causes["totaux"])
        self.assertEqual(sum(totaux), 89)

    def test_aucune_variante_du_pretraitement_ne_passe_le_seuil(self):
        etude = donnees_resultats.PRETRAITEMENT
        self.assertEqual(len(etude["variantes"]), 18)
        self.assertTrue(all(ecart < etude["seuil"] for _, ecart in etude["variantes"]))

    def test_langues_par_methode(self):
        langues = donnees_resultats.PAR_LANGUE
        self.assertEqual(sum(nb for _, _, nb in langues["colonnes"]), 63)
        somali = {ligne["nom"]: ligne["valeurs"][2] for ligne in langues["lignes"]}
        self.assertGreater(somali["TF-IDF"], somali["e5-small"], "en somali, TF-IDF devance e5-small")
        self.assertGreater(somali["e5-small"], somali["LaBSE"])


# ----------------------------------------------------------------------------------------------- questions prêtes et cache
class QuestionsPretesTests(IsoleTestCase):
    def fichier_requetes(self, lignes):
        dossier = tempfile.TemporaryDirectory()
        self.addCleanup(dossier.cleanup)
        chemin = Path(dossier.name) / "requetes.csv"
        chemin.write_text("id,requete,langue,type,faits,partition\n" + "\n".join(lignes) + "\n", encoding="utf-8")
        return str(chemin)

    def test_service_infidele_ne_remplit_jamais_le_cache(self):
        self.avec_jeton_et_faux_client(FauxClient())                     # vecteurs au hasard : pas ceux du notebook
        sortie = StringIO()
        call_command("preparer_en_ligne", stdout=sortie)
        self.assertIn("NON mises en cache", sortie.getvalue())
        self.assertEqual(moteur.lire_cache(), {})
        self.assertEqual(moteur.questions_pretes(), [])

    def test_exemples_prets_apres_une_preparation_fidele(self):
        self.avec_jeton_et_faux_client(FauxClient(fidele=True))
        call_command("preparer_en_ligne", stdout=StringIO())
        pretes = moteur.questions_pretes()
        self.assertEqual(len(pretes), 6)
        self.assertEqual({q["langue"] for q in pretes}, {"fr", "en", "so"})
        self.assertEqual({q["type"] for q in pretes}, {"exemple", "cas difficile"})

    def test_evaluation_ajoute_les_requetes_du_fichier(self):
        fichier = self.fichier_requetes(["Q1,Où est la bibliothèque ?,fr,mots_communs,horaires_bibliotheque,test",
                                         "Q2,Where is the library?,en,mots_communs,horaires_bibliotheque,dev",
                                         "Q3,  Maktabaddu   waa   furan ,so,court,horaires_bibliotheque,test",
                                         "Q4,Combien coûtent les frais de scolarité en Master ?,fr,mots_communs,montant_frais,test"])
        self.avec_jeton_et_faux_client(FauxClient(fidele=True))
        sortie = StringIO()
        call_command("preparer_en_ligne", "--evaluation", f"--fichier-requetes={fichier}", stdout=sortie)
        pretes = {q["requete"]: q for q in moteur.questions_pretes()}
        self.assertEqual(len(pretes), 9, "6 exemples + 3 requêtes nouvelles (Q4 est déjà un exemple)")
        self.assertEqual(pretes["Où est la bibliothèque ?"]["type"], "évaluation")
        self.assertIn("Maktabaddu waa furan", pretes, "espaces nettoyés comme dans la recherche")
        self.assertEqual(pretes["Where is the library?"]["langue"], "en")
        self.assertIn("Tout est prêt", sortie.getvalue())

    def test_fichier_de_requetes_introuvable(self):
        self.avec_jeton_et_faux_client(FauxClient(fidele=True))
        with self.assertRaisesMessage(CommandError, "introuvable"):
            call_command("preparer_en_ligne", "--evaluation", "--fichier-requetes=/n-existe-pas.csv", stdout=StringIO())

    def test_une_question_n_est_prete_que_si_tous_les_modeles_l_ont(self):
        moteur.ajouter_au_cache("embeddings", "seulement e5", np.ones(384))
        moteur.FICHIER_QUESTIONS.write_text(json.dumps([{"requete": "seulement e5", "langue": "fr", "type": "exemple"}]), encoding="utf-8")
        self.assertEqual(moteur.questions_pretes(), [])
        moteur.ajouter_au_cache("labse", "seulement e5", np.ones(768))
        self.assertEqual(len(moteur.questions_pretes()), 1)

    def test_sans_service_ni_cache_message_clair(self):
        html = self.client.get(reverse("accueil")).content.decode()
        self.assertEqual(html.count("service non branché"), 4)
        self.assertNotIn("indisponible ici", html)
        self.assertIn("aucun service d'embeddings n'est branché sur ce site", html)
        self.assertNotIn("questions prêtes pour toutes les méthodes", html)

    def test_site_avec_cache_seulement(self):
        """Comme le site en ligne sans service : les questions préparées marchent, les autres retombent sur TF-IDF."""
        self.avec_jeton_et_faux_client(FauxClient(fidele=True))
        call_command("preparer_en_ligne", stdout=StringIO())
        os.environ.pop("HF_TOKEN")
        self.reinitialiser_caches()
        html = self.client.get(reverse("accueil")).content.decode()
        self.assertEqual(html.count("questions prêtes seulement"), 4)
        self.assertIn("6 questions prêtes pour toutes les méthodes", html)
        self.assertIn("marchent pour les <strong>6 questions prêtes</strong>", html)
        self.assertNotIn("disabled", html.split('<fieldset class="methodes">')[1].split("</fieldset>")[0])
        prete = views.EXEMPLES_QUI_MARCHENT[0]
        reponse = self.client.get(reverse("accueil"), {"q": prete, "methode": "melange"})
        self.assertNotContains(reponse, "pas disponible sur cet ordinateur")
        self.assertContains(reponse, "Mélange TF-IDF + e5-small")
        reponse = self.client.get(reverse("accueil"), {"q": "une question jamais préparée", "methode": "melange"})
        self.assertContains(reponse, "pas disponible sur cet ordinateur")
        self.assertContains(reponse, "Les résultats ci-dessous viennent de la méthode TF-IDF")

    def test_les_questions_prretes_sont_des_liens_de_recherche(self):
        self.avec_jeton_et_faux_client(FauxClient(fidele=True))
        call_command("preparer_en_ligne", stdout=StringIO())
        html = self.client.get(reverse("accueil")).content.decode()
        self.assertIn("pretes__liste", html)
        self.assertIn(f"?q={quote(views.EXEMPLES_QUI_MARCHENT[1])}", html)


class CacheFourniTests(SimpleTestCase):
    """Le cache publié avec le site (vecteurs de 83 questions, calculés par le serveur Colab vérifié) doit redonner exactement
    les résultats du notebook, sans aucun service en ligne."""

    def setUp(self):
        environnement = mock.patch.dict(os.environ)
        environnement.start()
        self.addCleanup(environnement.stop)
        for variable in ("HF_TOKEN", "PROJET8_MODE", "PROJET8_SERVEUR", "PROJET8_CLE"):
            os.environ.pop(variable, None)
        patch = mock.patch.object(moteur, "modele_local_pret", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)
        moteur._cache["charge"] = None
        moteur.vecteur_requete.cache_clear()
        self.addCleanup(lambda: (moteur._cache.update(charge=None), moteur.vecteur_requete.cache_clear()))

    def test_cache_fourni_reproduit_les_resultats_du_notebook(self):
        requetes = RACINE_PROJET / "data" / "requetes.csv"
        if not (requetes.exists() and moteur.FICHIER_CACHE.exists() and moteur.FICHIER_QUESTIONS.exists()):
            self.skipTest("cache ou jeu de requêtes absent")
        import pandas as pd
        test = pd.read_csv(requetes, encoding="utf-8-sig").fillna("").query("partition == 'test'")
        faits = [set(f.split(";")) if f else set() for f in moteur.charger_passages()["faits"]]
        attendu = {"tfidf": (50, 0.708), "embeddings": (50, 0.702), "labse": (36, 0.523), "melange": (53, 0.792), "melange_labse": (52, 0.742)}
        self.assertEqual(len(moteur.questions_pretes()), 83)
        for methode, (nb_reussies, mrr) in attendu.items():
            hits, rrs = 0, []
            for _, r in test.iterrows():
                bonnes = {i for i, f in enumerate(faits) if f & set(r["faits"].split(";"))}
                classement = moteur.classer(moteur.scores(moteur.nettoyer(r["requete"]), methode))
                rang = next((k + 1 for k, i in enumerate(classement) if i in bonnes), None)
                hits += bool(rang and rang <= 3)
                rrs.append(1 / rang if rang else 0.0)
            self.assertEqual(hits, nb_reussies, f"{methode} : requêtes réussies en top 3")
            self.assertAlmostEqual(float(np.mean(rrs)), mrr, places=3, msg=f"{methode} : MRR")

    def test_les_questions_pretes_sont_toutes_utilisables_sans_service(self):
        if not moteur.FICHIER_QUESTIONS.exists():
            self.skipTest("liste absente")
        for q in moteur.questions_pretes()[:10]:
            for methode in ("embeddings", "labse", "melange", "melange_labse"):
                self.assertEqual(len(moteur.rechercher(q["requete"], methode)), 3, (q["requete"], methode))
