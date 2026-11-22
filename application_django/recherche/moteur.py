"""Moteur de recherche du projet 8 (recherche sémantique dans des documents universitaires).

Il reprend les méthodes du notebook à partir des fichiers du dossier `artefacts/`, produits par le notebook sur Colab :
  - tfidf          : TF-IDF (n-grammes de caractères 3 à 5) + similarité cosinus (baseline) ;
  - embeddings     : multilingual-e5-small ;
  - labse          : LaBSE ;
  - melange        : alpha x TF-IDF + (1 - alpha) x e5-small, alpha = 0,5 ;
  - melange_labse  : même principe avec LaBSE, alpha = 0,25.
Les valeurs d'alpha sont celles choisies sur les requêtes de réglage (dev) dans le notebook.

Les vecteurs des PASSAGES sont déjà calculés (fichiers .npy). Pour une méthode à embeddings, il ne reste qu'à calculer le
vecteur de la QUESTION, de trois façons :
  - serveur : le serveur d'embeddings du projet, lancé sur Colab (notebook Serveur_Embeddings_Colab.ipynb), dont l'adresse et
    la clé sont lues dans les variables d'environnement PROJET8_SERVEUR et PROJET8_CLE ;
  - en ligne : API d'inférence de Hugging Face, avec un jeton lu dans la variable d'environnement HF_TOKEN ;
  - en local : bibliothèque sentence-transformers et modèle déjà téléchargé (jamais de téléchargement silencieux).
Dans les deux premiers cas, seul le texte de la question est envoyé, jamais les documents.
La variable PROJET8_MODE vaut « auto » (par défaut : serveur si PROJET8_SERVEUR existe, sinon en ligne si HF_TOKEN existe,
sinon local), « serveur », « en_ligne » ou « local ».
Les vecteurs de questions obtenus en ligne sont gardés dans un fichier de cache : une démonstration préparée avec internet
peut ensuite être rejouée sans connexion. Sans aucun de ces moyens, l'application le dit et TF-IDF reste disponible.
"""
import http.client
import importlib.util
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOSSIER_ARTEFACTS = Path(__file__).resolve().parent.parent / "artefacts"
FICHIER_CACHE = DOSSIER_ARTEFACTS / "cache_vecteurs_requetes.json"
FICHIER_QUESTIONS = DOSSIER_ARTEFACTS / "questions_pretes.json"     # questions dont les vecteurs sont dans le cache (écrit par preparer_en_ligne)
DELAI_EN_LIGNE = 30          # secondes d'attente maximale pour une réponse du service en ligne
AGENT_HTTP = "projet8-recherche/1.0"     # identifiant générique envoyé au serveur Colab (aucune donnée personnelle)
K = 3

METHODES = {
    "tfidf": "TF-IDF (baseline)",
    "embeddings": "Embeddings e5-small",
    "labse": "Embeddings LaBSE",
    "melange": "Mélange TF-IDF + e5-small",
    "melange_labse": "Mélange TF-IDF + LaBSE",
}
# méthode d'embeddings -> (nom du modèle, préfixe de requête demandé par sa fiche, préfixe de passage)
MODELES = {
    "embeddings": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "labse": ("sentence-transformers/LaBSE", "", ""),
}
ALPHAS = {"melange": 0.5, "melange_labse": 0.25}
DENSE_DE = {"melange": "embeddings", "melange_labse": "labse"}   # méthode de mélange -> méthode d'embeddings


class MethodeIndisponible(Exception):
    """La méthode demandée ne peut pas tourner pour l'instant (pas de jeton, de modèle local ou de connexion)."""


def nettoyer(texte):
    """Nettoyage léger identique à celui du notebook : Unicode NFC et espaces propres."""
    texte = unicodedata.normalize("NFC", str(texte))
    return re.sub(r"\s+", " ", texte).strip()


@lru_cache(maxsize=1)
def charger_passages():
    return pd.read_csv(DOSSIER_ARTEFACTS / "passages.csv", encoding="utf-8").fillna("")


@lru_cache(maxsize=1)
def charger_tfidf():
    """Vectoriseur et matrice TF-IDF. On lit les fichiers du notebook ; s'ils sont illisibles (version de scikit-learn
    différente de celle de Colab), on recalcule la même chose à partir des passages (même réglages)."""
    passages = charger_passages()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec = joblib.load(DOSSIER_ARTEFACTS / "tfidf_vectoriseur.joblib")
            mat = joblib.load(DOSSIER_ARTEFACTS / "tfidf_matrice.joblib")
        if mat.shape[0] != len(passages):
            raise ValueError("matrice et passages incohérents")
        return vec, mat, "fichiers du notebook"
    except Exception:
        vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                              analyzer="char_wb", ngram_range=(3, 5))
        return vec, vec.fit_transform(passages["texte"]), "recalculé à partir des passages"


# ----------------------------------------------------------------------------------------------- cache des questions
_cache = {"charge": None}


def lire_cache():
    """Vecteurs de questions déjà calculés en ligne : {« méthode|question » : vecteur}."""
    if _cache["charge"] is None:
        try:
            _cache["charge"] = json.loads(FICHIER_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache["charge"] = {}
    return _cache["charge"]


def ajouter_au_cache(methode, requete, vecteur):
    cache = lire_cache()
    cache[f"{methode}|{requete}"] = [round(float(x), 6) for x in vecteur]
    try:
        FICHIER_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass                                   # le cache est un confort : on ne bloque pas la recherche


def requetes_en_cache(methode):
    return [cle.split("|", 1)[1] for cle in lire_cache() if cle.startswith(f"{methode}|")]


def questions_pretes():
    """Questions [{requete, langue, type}] dont le vecteur est en cache pour TOUS les modèles : elles marchent avec toutes les
    méthodes à embeddings, même quand aucun service n'est branché."""
    try:
        liste = json.loads(FICHIER_QUESTIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    cache = lire_cache()
    return [q for q in liste if all(f"{m}|{q['requete']}" in cache for m in MODELES)]


# ----------------------------------------------------------------------------------------------- modes d'embeddings
def jeton_hf():
    return os.environ.get("HF_TOKEN", "").strip()


def adresse_serveur():
    """Adresse du serveur d'embeddings Colab (variable PROJET8_SERVEUR), sans « / » final ; vide si elle n'est pas définie."""
    return os.environ.get("PROJET8_SERVEUR", "").strip().rstrip("/")


def cle_serveur():
    return os.environ.get("PROJET8_CLE", "").strip()


def modele_telecharge(nom_modele):
    """Vrai si le modèle est déjà dans le cache local de Hugging Face (aucun accès réseau)."""
    racine = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (racine / ("models--" + nom_modele.replace("/", "--"))).exists()


def modele_local_pret(nom_modele):
    return importlib.util.find_spec("sentence_transformers") is not None and modele_telecharge(nom_modele)


def mode_embeddings(methode_dense):
    """« serveur », « en_ligne », « local » ou None : comment calculer le vecteur d'une question pour ce modèle."""
    choix = os.environ.get("PROJET8_MODE", "auto")
    if choix in ("auto", "serveur") and adresse_serveur():
        return "serveur"
    if choix in ("auto", "en_ligne") and jeton_hf():
        return "en_ligne"
    if choix in ("auto", "local") and modele_local_pret(MODELES[methode_dense][0]):
        return "local"
    return None


def disponibilite(methode):
    """(disponible, remarque). Une remarque non vide avec disponible = vrai signale un fonctionnement partiel."""
    if methode not in METHODES:
        return False, "méthode inconnue"
    if methode == "tfidf":
        return True, ""
    dense = DENSE_DE.get(methode, methode)
    if not (DOSSIER_ARTEFACTS / f"{dense}_passages.npy").exists():
        return False, "fichier d'embeddings absent"
    mode = mode_embeddings(dense)
    if mode == "serveur":
        return True, "serveur Colab"
    if mode == "en_ligne":
        return True, "en ligne"
    if mode == "local":
        return True, ""
    if requetes_en_cache(dense):
        return True, "questions déjà calculées seulement"
    return False, "ni serveur Colab (PROJET8_SERVEUR), ni jeton Hugging Face (HF_TOKEN), ni modèle local"


def service_en_ligne():
    """« serveur » (Colab), « huggingface » ou None : le service qui calcule les vecteurs des questions."""
    for m in MODELES:
        mode = mode_embeddings(m)
        if mode == "serveur":
            return "serveur"
        if mode == "en_ligne":
            return "huggingface"
    return None


def mode_en_ligne_actif():
    return service_en_ligne() is not None


@lru_cache(maxsize=None)
def matrice_dense(methode):
    return np.load(DOSSIER_ARTEFACTS / f"{methode}_passages.npy")


def message_erreur_en_ligne(erreur):
    texte = str(erreur)
    jeton = jeton_hf()
    if jeton:
        texte = texte.replace(jeton, "***")
    minuscule = texte.lower()
    if "401" in texte or "403" in texte or "unauthorized" in minuscule or "forbidden" in minuscule:
        return "jeton Hugging Face refusé (HF_TOKEN invalide ou sans droit d'inférence)"
    if "404" in texte or "not found" in minuscule or "doesn't support task" in minuscule:
        return "ce modèle n'est pas servi par le service en ligne (erreur 404)"
    if "429" in texte or "rate limit" in minuscule:
        return "limite de requêtes du service en ligne atteinte"
    if "402" in texte or "payment required" in minuscule or "credit" in minuscule or "crédit" in minuscule:
        return "crédits Hugging Face épuisés ou service payant pour ce compte (erreur 402)"
    if "timeout" in minuscule or "timed out" in minuscule:
        return "le service en ligne n'a pas répondu à temps"
    if "connection" in minuscule or "resolve" in minuscule or "network" in minuscule:
        return "pas de connexion internet"
    return f"{type(erreur).__name__} : {texte[:200]}"


def url_service(nom_modele):
    """Adresse du service d'embeddings pour un modèle. On la passe en entier à la bibliothèque parce que certains modèles,
    dont multilingual-e5-small, sont enregistrés chez Hugging Face pour la tâche « similarité de phrases » : la bibliothèque
    refuserait la tâche « extraction de vecteurs » alors que le service les propose avec la même adresse de base."""
    return f"https://router.huggingface.co/hf-inference/models/{nom_modele}/pipeline/feature-extraction"


def client_hf():
    from huggingface_hub import InferenceClient
    return InferenceClient(provider="hf-inference", api_key=jeton_hf(), timeout=DELAI_EN_LIGNE)


def vecteur_en_ligne(methode, texte):
    """Vecteur d'un texte (préfixe déjà ajouté) calculé par l'API d'inférence de Hugging Face, puis normalisé."""
    nom_modele = MODELES[methode][0]
    brut = None
    for tentative in range(2):
        try:
            brut = client_hf().feature_extraction(texte, model=url_service(nom_modele))
            break
        except Exception as erreur:                                      # réseau, jeton, modèle absent...
            if tentative == 0 and ("503" in str(erreur) or "loading" in str(erreur).lower()):
                time.sleep(4)                                            # modèle en cours de chargement : on réessaie
                continue
            raise MethodeIndisponible(message_erreur_en_ligne(erreur)) from erreur
    return valider_vecteur(methode, brut)


def valider_vecteur(methode, brut):
    """Vérifie qu'un service a renvoyé UN vecteur de la bonne dimension (donc le bon modèle), puis le normalise."""
    v = np.asarray(brut, dtype=float)
    if v.ndim == 2 and v.shape[0] == 1:
        v = v[0]
    if v.ndim != 1:
        raise MethodeIndisponible("le service a renvoyé un vecteur par mot au lieu d'un vecteur par phrase")
    attendu = matrice_dense(methode).shape[1]
    if v.shape[0] != attendu:
        raise MethodeIndisponible(f"dimension inattendue ({v.shape[0]} au lieu de {attendu}) : ce n'est pas le bon modèle")
    norme = np.linalg.norm(v)
    if norme == 0:
        raise MethodeIndisponible("vecteur nul renvoyé par le service")
    return v / norme


def message_erreur_serveur(code, detail=""):
    """Message clair pour une erreur du serveur Colab (code HTTP, ou None si le serveur n'a pas répondu du tout)."""
    texte = detail or ""
    if cle_serveur():
        texte = texte.replace(cle_serveur(), "***")
    if code == 401:
        return "clé du serveur Colab refusée (PROJET8_CLE n'est pas celle affichée par Colab)"
    if code in (400, 413, 500):
        return f"le serveur Colab a refusé la demande (erreur {code} : {texte[:150]})"
    if code == 403:
        return "accès refusé par le tunnel ou le serveur Colab (erreur 403)"
    if code in (404, 502, 503, 504) or (code is not None and code >= 520):
        return ("serveur Colab injoignable : la cellule Colab est arrêtée ou son adresse a changé "
                f"(erreur {code}) ; la relancer puis mettre à jour PROJET8_SERVEUR et PROJET8_CLE")
    if code is not None:
        return f"erreur {code} du serveur Colab"
    if "timed out" in texte.lower() or "timeout" in texte.lower():
        return "le serveur Colab n'a pas répondu à temps"
    return ("serveur Colab injoignable (pas de connexion internet, cellule Colab arrêtée ou adresse changée) ; "
            "vérifier PROJET8_SERVEUR")


def vecteur_serveur(methode, texte):
    """Vecteur d'un texte (préfixe déjà ajouté) calculé par le serveur Colab du projet, puis normalisé."""
    adresse = adresse_serveur()
    if not adresse.startswith(("http://", "https://")):
        raise MethodeIndisponible("PROJET8_SERVEUR doit être une adresse complète (https://...trycloudflare.com)")
    corps = json.dumps({"modele": MODELES[methode][0], "textes": [texte]}).encode("utf-8")
    requete = urllib.request.Request(
        adresse + "/embeddings", data=corps, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + cle_serveur(), "User-Agent": AGENT_HTTP})
    try:
        with urllib.request.urlopen(requete, timeout=DELAI_EN_LIGNE) as reponse:
            brut = json.loads(reponse.read().decode("utf-8"))["vecteurs"]
    except urllib.error.HTTPError as erreur:                         # le serveur (ou le tunnel) a répondu par une erreur
        try:
            detail = json.loads(erreur.read().decode("utf-8")).get("erreur", "")
        except (OSError, ValueError, AttributeError):
            detail = ""
        raise MethodeIndisponible(message_erreur_serveur(erreur.code, detail)) from erreur
    except (KeyError, TypeError, ValueError) as erreur:               # réponse qui n'est pas celle du serveur du projet
        raise MethodeIndisponible("réponse illisible : l'adresse PROJET8_SERVEUR n'est pas celle du serveur du projet") from erreur
    except (OSError, http.client.HTTPException) as erreur:            # pas de réseau, délai dépassé, adresse introuvable
        raise MethodeIndisponible(message_erreur_serveur(None, str(erreur))) from erreur
    return valider_vecteur(methode, brut)


def vecteur_service(methode, texte):
    """Vecteur d'un texte (préfixe déjà ajouté) calculé par le service en ligne actif : serveur Colab ou Hugging Face."""
    mode = mode_embeddings(methode)
    if mode == "serveur":
        return vecteur_serveur(methode, texte)
    if mode == "en_ligne":
        return vecteur_en_ligne(methode, texte)
    raise MethodeIndisponible("aucun service en ligne configuré (PROJET8_SERVEUR ou HF_TOKEN)")


@lru_cache(maxsize=None)
def charger_modele_local(nom_modele):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nom_modele, local_files_only=True)


@lru_cache(maxsize=512)
def vecteur_requete(methode, requete):
    """Vecteur normalisé d'une question : cache, puis en ligne ou en local selon le mode."""
    cache = lire_cache()
    cle = f"{methode}|{requete}"
    if cle in cache:
        return np.asarray(cache[cle], dtype=float)
    nom_modele, prefixe_requete, _ = MODELES[methode]
    mode = mode_embeddings(methode)
    if mode in ("serveur", "en_ligne"):
        v = vecteur_service(methode, prefixe_requete + requete)
        ajouter_au_cache(methode, requete, v)
        return v
    if mode == "local":
        modele = charger_modele_local(nom_modele)
        return modele.encode([prefixe_requete + requete], normalize_embeddings=True, show_progress_bar=False)[0]
    raise MethodeIndisponible("ni serveur Colab (PROJET8_SERVEUR), ni jeton Hugging Face (HF_TOKEN), ni modèle local, "
                              "et question absente du cache")


# ----------------------------------------------------------------------------------------------- scores et recherche
def scores_tfidf(requete):
    vec, mat, _ = charger_tfidf()
    return cosine_similarity(vec.transform([requete]), mat).ravel()


def scores_dense(requete, methode):
    ok, raison = disponibilite(methode)
    if not ok:
        raise MethodeIndisponible(raison)
    try:
        return matrice_dense(methode) @ vecteur_requete(methode, requete)   # vecteurs normalisés : produit scalaire = cosinus
    except MethodeIndisponible:
        raise
    except Exception as erreur:                                              # modèle local illisible, mémoire insuffisante...
        raise MethodeIndisponible(f"{type(erreur).__name__} : {erreur}") from erreur


def minmax(x):
    x = np.asarray(x, dtype=float)
    ecart = x.max() - x.min()
    return (x - x.min()) / ecart if ecart > 0 else np.zeros_like(x)


def scores(requete, methode):
    if methode == "tfidf":
        return scores_tfidf(requete)
    if methode in MODELES:
        return scores_dense(requete, methode)
    if methode in ALPHAS:
        a = ALPHAS[methode]
        return a * minmax(scores_tfidf(requete)) + (1 - a) * minmax(scores_dense(requete, DENSE_DE[methode])) + 1e-9
    raise ValueError(f"méthode inconnue : {methode}")


def classer(valeurs):
    """Indices des passages par score décroissant. Les scores nuls sont écartés (pas de classement au hasard)."""
    ordre = np.argsort(-valeurs, kind="stable")
    return [int(i) for i in ordre if valeurs[i] > 0]


def corps_du_passage(ligne):
    """Texte du passage sans son titre (le titre est affiché à part)."""
    texte, titre = ligne["texte"], ligne["titre"]
    return texte[len(titre):].strip() if texte.startswith(titre) else texte


def rechercher(requete, methode="tfidf", k=K):
    """Les k meilleurs passages pour une requête : liste de dictionnaires."""
    requete = nettoyer(requete)
    passages = charger_passages()
    valeurs = scores(requete, methode)
    resultats = []
    for rang, i in enumerate(classer(valeurs)[:k], start=1):
        ligne = passages.iloc[i]
        resultats.append({
            "rang": rang,
            "score": round(float(valeurs[i]), 3),
            "id": ligne["id"],
            "source": ligne["source"],
            "langue": ligne["langue"],
            "titre": ligne["titre"],
            "passage": corps_du_passage(ligne),
        })
    return resultats


def reponse_api(requete, methode="tfidf", k=K):
    """Même résultat sous forme de dictionnaire (comme la fonction du notebook), pour l'API JSON."""
    return {"requete": nettoyer(requete), "methode": methode, "resultats": rechercher(requete, methode, k)}


# ----------------------------------------------------------------------------------------------- aides pour l'interface
# Surlignage des mots communs, carte des passages et statistiques du corpus : ces fonctions ne servent qu'à l'affichage,
# elles ne changent jamais le classement des résultats.
MOT_RE = re.compile(r"[^\W\d_]+", re.UNICODE)                                    # sous-mots : « l'année » donne « l » et « année »
MOTS_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+(?:[.,]\d+)?", re.UNICODE)   # mots du notebook (regex adaptée, sans ponctuation)
SEUIL_FORME_FREQUENTE = 1 / 3       # une forme présente dans au moins un tiers des passages d'une langue n'est pas surlignée
LONGUEUR_MIN_MOT = 3


def normaliser(mot):
    """Minuscules et sans accents : « Scolarité » donne « scolarite »."""
    decompose = unicodedata.normalize("NFD", mot.lower())
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=1)
def formes_frequentes():
    """{langue : formes (sans accents) présentes dans au moins un tiers des passages de cette langue (mots outils)}."""
    resultat = {}
    for langue, groupe in charger_passages().groupby("langue"):
        compte = {}
        for texte in groupe["texte"]:
            for forme in {normaliser(m) for m in MOT_RE.findall(texte)}:
                compte[forme] = compte.get(forme, 0) + 1
        resultat[langue] = {forme for forme, n in compte.items() if n / len(groupe) >= SEUIL_FORME_FREQUENTE}
    return resultat


def formes_de_la_requete(requete):
    """Formes de la question (sans accents, 3 lettres au moins) qui peuvent être surlignées dans un passage."""
    return {normaliser(m) for m in MOT_RE.findall(requete) if len(m) >= LONGUEUR_MIN_MOT}


def decouper_surlignage(texte, formes_requete, langue):
    """Découpe un passage en morceaux [(texte, surligné)] et donne la liste des mots communs avec la question.
    Un mot est surligné s'il est dans la question et n'est pas un mot très fréquent de la langue du passage."""
    utiles = formes_requete - formes_frequentes().get(langue, set())
    morceaux, position, communs = [], 0, []
    for m in MOT_RE.finditer(texte):
        if normaliser(m.group(0)) not in utiles:
            continue
        if m.start() > position:
            morceaux.append((texte[position:m.start()], False))
        morceaux.append((m.group(0), True))
        position = m.end()
        mot = m.group(0).lower()
        if mot not in communs:
            communs.append(mot)
    if position < len(texte):
        morceaux.append((texte[position:], False))
    return morceaux, communs


@lru_cache(maxsize=1)
def base_carte():
    """Projection PCA (2 axes) des vecteurs e5 des passages : (moyenne, axes, échelle, coordonnées, part de variance)."""
    X = matrice_dense("embeddings").astype(float)
    moyenne = X.mean(axis=0)
    _, valeurs, Vt = np.linalg.svd(X - moyenne, full_matrices=False)
    axes = Vt[:2]
    signes = np.sign(axes[np.arange(2), np.abs(axes).argmax(axis=1)])           # signe fixé : la carte ne « se retourne » pas
    axes = axes * signes[:, None]
    coordonnees = (X - moyenne) @ axes.T
    echelle = np.abs(coordonnees).max(axis=0)                                    # chaque axe remplit la carte
    variance = (valeurs[:2] ** 2) / (valeurs ** 2).sum()
    return moyenne, axes, echelle, coordonnees / echelle, variance


def carte_passages():
    """Données de la carte : un point par passage (x et y entre -1 et 1), avec sa langue, son titre et son document."""
    passages = charger_passages()
    _, _, _, coordonnees, variance = base_carte()
    points = [{"id": ligne["id"], "x": round(float(coordonnees[i][0]), 4), "y": round(float(coordonnees[i][1]), 4),
               "l": ligne["langue"], "t": ligne["titre"], "d": ligne["source"].removesuffix(".txt")}
              for i, (_, ligne) in enumerate(passages.iterrows())]
    return {"points": points, "variance": round(float(variance.sum()) * 100)}


def position_requete(vecteur):
    """Position d'un vecteur de question e5 sur la carte des passages."""
    moyenne, axes, echelle, _, _ = base_carte()
    p = (np.asarray(vecteur, dtype=float) - moyenne) @ axes.T / echelle
    return [round(float(np.clip(p[0], -1.2, 1.2)), 4), round(float(np.clip(p[1], -1.2, 1.2)), 4)]


@lru_cache(maxsize=1)
def statistiques_corpus():
    """Documents (langue, passages, mots, titres des passages) et totaux par langue, calculés sur passages.csv."""
    passages = charger_passages()
    documents = []
    for source, groupe in passages.groupby("source", sort=False):
        premier = groupe.iloc[0]["titre"]
        documents.append({
            "source": source, "nom": source.removesuffix(".txt"), "langue": groupe.iloc[0]["langue"],
            "titre": premier.capitalize() if premier.isupper() else premier,
            "passages": len(groupe), "mots": int(sum(len(MOTS_RE.findall(t)) for t in groupe["texte"])),
            "sections": [{"id": ligne["id"], "titre": ligne["titre"]} for _, ligne in groupe.iterrows()],
        })
    langues = {}
    for d in documents:
        total = langues.setdefault(d["langue"], {"documents": 0, "passages": 0, "mots": 0})
        total["documents"] += 1
        total["passages"] += d["passages"]
        total["mots"] += d["mots"]
    return {"documents": documents, "langues": langues, "passages": len(passages),
            "mots": sum(v["mots"] for v in langues.values())}
