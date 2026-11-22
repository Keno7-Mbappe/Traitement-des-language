import json
import time

from django.http import JsonResponse
from django.shortcuts import render
from django.utils.html import conditional_escape, format_html
from django.utils.safestring import mark_safe

from . import donnees_resultats, moteur

LONGUEUR_MAX = 300
LANGUES = {"fr": "Français", "en": "English", "so": "Soomaali"}
EXEMPLES_QUI_MARCHENT = [
    "Combien coûtent les frais de scolarité en Master ?",
    "Do I need a visa to study here?",
    "Maktabaddu goorma ayay furan tahay?",
]
EXEMPLES_DIFFICILES = [      # cas où TF-IDF échoue (voir l'analyse d'erreurs du notebook)
    "Je n'ai plus d'argent pour payer ma scolarité, que faire ?",
    "I cannot afford my tuition at the moment, what are my options?",
    "Sideen u codsadaa deeq waxbarasho?",
]

# Présentation des méthodes : nom court, ligne de détail, étiquette éventuelle, étapes du traitement de la question.
PRESENTATION = {
    "tfidf": {"court": "TF-IDF", "detail": "compare les lettres", "etiquette": "baseline",
              "etapes": ["Question", "Nettoyage", "N-grammes de lettres", "Cosinus TF-IDF", "Top 3"]},
    "embeddings": {"court": "e5-small", "detail": "compare le sens", "etiquette": "",
                   "etapes": ["Question", "Nettoyage", "Vecteur e5 (384)", "Cosinus", "Top 3"]},
    "labse": {"court": "LaBSE", "detail": "sens, entre langues", "etiquette": "",
              "etapes": ["Question", "Nettoyage", "Vecteur LaBSE (768)", "Cosinus", "Top 3"]},
    "melange": {"court": "Mélange e5", "detail": "lettres + sens", "etiquette": "meilleur",
                "etapes": ["Question", "Nettoyage", "TF-IDF + vecteur e5", "Mélange α = 0,5", "Top 3"]},
    "melange_labse": {"court": "Mélange LaBSE", "detail": "lettres + sens", "etiquette": "",
                      "etapes": ["Question", "Nettoyage", "TF-IDF + vecteur LaBSE", "Mélange α = 0,25", "Top 3"]},
}


def lire_parametres(request):
    """Requête et méthode demandées dans l'adresse (q et methode). Retourne (requete, methode, erreur)."""
    requete = moteur.nettoyer(request.GET.get("q", ""))
    methode = request.GET.get("methode", "tfidf")
    if methode not in moteur.METHODES:
        return requete, methode, f"méthode inconnue : {methode}"
    if len(requete) > LONGUEUR_MAX:
        return requete, methode, f"requête trop longue (maximum {LONGUEUR_MAX} caractères)"
    return requete, methode, ""


def passage_en_html(morceaux):
    """Passage découpé par moteur.decouper_surlignage : le texte est échappé, seuls les mots communs reçoivent <mark>."""
    return mark_safe("".join(format_html("<mark>{}</mark>", texte) if surligne else conditional_escape(texte)
                             for texte, surligne in morceaux))


def preparer_resultats(requete, resultats):
    """Ajoute à chaque résultat le passage surligné, les mots communs avec la question et la part du meilleur score."""
    formes = moteur.formes_de_la_requete(requete)
    meilleur = max((r["score"] for r in resultats), default=0) or 1
    for r in resultats:
        morceaux, communs = moteur.decouper_surlignage(r["passage"], formes, r["langue"])
        r.update(langue_nom=LANGUES.get(r["langue"], r["langue"]), passage_html=passage_en_html(morceaux),
                 communs=communs, part=max(4, round(100 * r["score"] / meilleur)))


def donnees_carte(requete, methode_utilisee, resultats):
    """Données de la carte des passages (projection PCA des vecteurs e5) avec les résultats et, si elle est connue, la question."""
    try:
        carte = moteur.carte_passages()
    except (OSError, ValueError, KeyError):
        return None
    carte["resultats"] = [r["id"] for r in resultats or []]
    carte["question"] = None
    if requete and methode_utilisee in ("embeddings", "melange"):          # le vecteur e5 de la question est déjà calculé
        try:
            carte["question"] = moteur.position_requete(moteur.vecteur_requete("embeddings", requete))
        except moteur.MethodeIndisponible:
            pass
    return carte


def pourcentage(valeur):
    return round(100 * valeur)


def chiffres_accueil():
    """Chiffres de la page d'accueil (corpus et résultats), tous tirés de moteur et de donnees_resultats."""
    par_cle = {m["cle"]: m for m in donnees_resultats.METHODES}
    meilleure, base, labse = par_cle["melange"]["toutes"], par_cle["tfidf"]["toutes"], par_cle["labse"]["toutes"]
    somali = {ligne["nom"]: ligne["valeurs"][2] for ligne in donnees_resultats.PAR_LANGUE["lignes"]}
    requetes_test = donnees_resultats.REQUETES_TEST["toutes"]
    return {
        "stats": moteur.statistiques_corpus(), "requetes_test": requetes_test,
        "meilleure_hit_pct": pourcentage(meilleure["hit"]), "base_hit_pct": pourcentage(base["hit"]),
        "meilleure_mrr": meilleure["mrr"], "base_mrr": base["mrr"],
        "ecart_requetes": round((meilleure["hit"] - base["hit"]) * requetes_test),
        "somali_tfidf_pct": pourcentage(somali["TF-IDF"]), "somali_e5_pct": pourcentage(somali["e5-small"]),
        "plus_gros_pct": pourcentage(labse["hit"]),
    }


def accueil(request):
    """Page de recherche : formulaire, exemples et top 3 des passages."""
    requete, methode, erreur = lire_parametres(request)
    if erreur and request.GET.get("q"):
        methode = methode if methode in moteur.METHODES else "tfidf"
    contexte = {
        "requete": requete,
        "methode": methode,
        "erreur": erreur if request.GET.get("q") else "",
        "exemples_ok": EXEMPLES_QUI_MARCHENT,
        "exemples_difficiles": EXEMPLES_DIFFICILES,
        "langues": LANGUES,
        "resultats": None,
        "avertissement": "",
        "methode_utilisee": methode,
    }

    if requete and not erreur:
        utilisee = methode
        try:
            debut = time.perf_counter()
            resultats = moteur.rechercher(requete, methode)
        except moteur.MethodeIndisponible as raison:
            contexte["avertissement"] = (
                f"La méthode « {moteur.METHODES[methode]} » n'est pas disponible sur cet ordinateur ({raison}). "
                "Les résultats ci-dessous viennent de la méthode TF-IDF."
            )
            utilisee = "tfidf"
            debut = time.perf_counter()
            resultats = moteur.rechercher(requete, "tfidf")
        preparer_resultats(requete, resultats)
        contexte.update(
            resultats=resultats,
            methode=utilisee,
            methode_utilisee=utilisee,
            duree_ms=round((time.perf_counter() - debut) * 1000),
            etapes=PRESENTATION[utilisee]["etapes"],
        )

    options = []
    for cle, libelle in moteur.METHODES.items():
        ok, raison = moteur.disponibilite(cle)
        options.append({"cle": cle, "libelle": libelle, "disponible": ok, "raison": raison, **PRESENTATION[cle]})
    contexte["options"] = options
    contexte["libelle_utilise"] = moteur.METHODES[contexte["methode_utilisee"]]
    contexte["carte"] = donnees_carte(requete if contexte["resultats"] is not None else "",
                                      contexte["methode_utilisee"], contexte["resultats"])
    if contexte["resultats"] is None:
        contexte.update(chiffres_accueil())
    return render(request, "recherche/accueil.html", contexte)


def exemple_api():
    """Vraie réponse de l'API pour une requête d'exemple (passage raccourci), affichée dans la page « Méthode »."""
    reponse = moteur.reponse_api("bibliotheque horaires", "tfidf", k=2)
    for r in reponse["resultats"]:
        r["passage"] = r["passage"][:84].rstrip() + "…"
    return json.dumps(reponse, ensure_ascii=False, indent=2)


def courbe_alpha():
    """Points de la courbe MRR(dev) selon α, pour chaque modèle : position (%) et hauteur (%) dans un graphique 0,5 à 0,8."""
    bas, haut = 0.5, 0.8
    courbes = []
    for nom, donnees in donnees_resultats.ALPHA.items():
        points = [{"alpha": a, "mrr": m, "hauteur": round(100 * (m - bas) / (haut - bas)), "choisi": a == donnees["choisi"]}
                  for a, m in donnees["valeurs"]]
        courbes.append({"nom": nom, "choisi": donnees["choisi"], "points": points})
    return courbes


def methode(request):
    """Page « Méthode » : la chaîne de traitement, les méthodes, les métriques et le protocole."""
    return render(request, "recherche/methode.html", {
        "stats": moteur.statistiques_corpus(),
        "methodes": [{"cle": cle, "libelle": libelle, **PRESENTATION[cle]} for cle, libelle in moteur.METHODES.items()],
        "fragmentation": donnees_resultats.FRAGMENTATION, "courbes_alpha": courbe_alpha(),
        "api_exemple": exemple_api(), "requetes_test": donnees_resultats.REQUETES_TEST,
    })


def barres_pretraitement():
    """Barres divergentes (écart de MRR dev par rapport à la référence) : position gauche et largeur en % d'un axe -0,22 à +0,06."""
    mini, maxi = -0.22, 0.06
    etendue = maxi - mini
    zero = 100 * (0 - mini) / etendue
    barres = []
    for nom, ecart in donnees_resultats.PRETRAITEMENT["variantes"]:
        largeur = 100 * abs(ecart) / etendue
        barres.append({"nom": nom, "ecart": ecart, "gauche": round(zero if ecart >= 0 else zero - largeur, 2),
                       "largeur": round(max(largeur, 0.6), 2)})          # même un écart nul reste visible
    return barres, round(zero, 2), round(100 * (donnees_resultats.PRETRAITEMENT["seuil"] - mini) / etendue, 2)


def barres_causes():
    """Une barre empilée par méthode : part de chaque cause parmi ses cas difficiles."""
    causes = donnees_resultats.CAUSES
    colonnes = []
    for j, nom in enumerate(causes["colonnes"]):
        total = causes["totaux"][j]
        segments = [{"nom": ligne["nom"], "valeur": ligne["valeurs"][j], "classe": i,
                     "pct": round(100 * ligne["valeurs"][j] / total, 2)}
                    for i, ligne in enumerate(causes["lignes"]) if ligne["valeurs"][j]]
        colonnes.append({"nom": nom, "total": total, "segments": segments})
    return colonnes


def resultats(request):
    """Page « Résultats » : comparaison des méthodes sur les 63 requêtes de test."""
    methodes = donnees_resultats.METHODES
    meilleure = next(m for m in methodes if m.get("meilleur"))
    base = next(m for m in methodes if m["cle"] == "tfidf")
    hasard = next(m for m in methodes if m["cle"] == "hasard")
    variantes = donnees_resultats.PRETRAITEMENT["variantes"]
    barres, zero, seuil = barres_pretraitement()
    langues = donnees_resultats.PAR_LANGUE
    meilleurs = [max(ligne["valeurs"][j] for ligne in langues["lignes"]) for j in range(len(langues["colonnes"]))]
    lignes_langues = [{"nom": ligne["nom"], "cases": [{"valeur": v, "meilleur": v == meilleurs[j]} for j, v in enumerate(ligne["valeurs"])]}
                      for ligne in langues["lignes"]]
    return render(request, "recherche/resultats.html", {
        "methodes": methodes, "requetes_test": donnees_resultats.REQUETES_TEST,
        "meilleure": meilleure, "base": base, "hasard": hasard,
        "ecart_requetes": round((meilleure["toutes"]["hit"] - base["toutes"]["hit"]) * donnees_resultats.REQUETES_TEST["toutes"]),
        "colonnes_langues": langues["colonnes"], "lignes_langues": lignes_langues,
        "entre_langues": donnees_resultats.ENTRE_LANGUES,
        "causes": donnees_resultats.CAUSES, "colonnes_causes": barres_causes(),
        "pretraitement": donnees_resultats.PRETRAITEMENT, "barres": barres, "zero": zero, "seuil": seuil,
        "meilleure_variante": max(variantes, key=lambda v: v[1]), "nb_variantes": len(variantes),
        "fragmentation": donnees_resultats.FRAGMENTATION,
    })


def corpus(request):
    """Page « Corpus » : les documents et leurs passages, calculés sur passages.csv."""
    stats = moteur.statistiques_corpus()
    plus_long = max(d["passages"] for d in stats["documents"])
    documents = [{**d, "part": round(100 * d["passages"] / plus_long), "langue_nom": LANGUES.get(d["langue"], d["langue"])}
                 for d in stats["documents"]]
    totaux = [{"code": code, "nom": LANGUES[code], **stats["langues"][code]} for code in ("fr", "en", "so") if code in stats["langues"]]
    return render(request, "recherche/corpus.html", {"stats": stats, "documents": documents, "totaux": totaux})


def api_recherche(request):
    """API JSON : /api/recherche/?q=...&methode=tfidf. Même structure que la fonction reponse_api du notebook."""
    requete, methode, erreur = lire_parametres(request)
    if not erreur and not requete:
        erreur = "paramètre q manquant"
    if erreur:
        return JsonResponse({"erreur": erreur}, status=400, json_dumps_params={"ensure_ascii": False})
    try:
        donnees = moteur.reponse_api(requete, methode)
    except moteur.MethodeIndisponible as raison:
        return JsonResponse({"erreur": f"méthode indisponible : {raison}", "methode": methode}, status=503,
                            json_dumps_params={"ensure_ascii": False})
    return JsonResponse(donnees, json_dumps_params={"ensure_ascii": False})


def page_introuvable(request, exception=None):
    return render(request, "404.html", status=404)


def erreur_serveur(request):
    return render(request, "500.html", status=500)
