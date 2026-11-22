"""Résultats du projet affichés par la page « Résultats ».

Les valeurs sont celles du notebook (exécution sur Colab, section 8 : 63 requêtes de test, K = 3). Elles sont écrites ici
une fois pour toutes : l'application n'évalue rien elle-même. Le test `ResultatsTests` compare le tableau principal au
README pour qu'ils ne se contredisent pas.
"""

# Pour chaque méthode : P@3, Hit@3 et MRR dans les deux scénarios. Hit@3 est écrit en fraction exacte (requêtes réussies sur
# 63 ou 54) : un arrondi à 3 décimales pourrait changer le dernier chiffre affiché.
METHODES = [
    {"cle": "hasard", "nom": "Hasard", "detail": "plancher : classement au hasard (valeurs exactes)",
     "toutes": {"p": 0.010, "hit": 0.029, "mrr": 0.047}, "meme": {"p": 0.004, "hit": 0.013, "mrr": 0.027}},
    {"cle": "tfidf", "nom": "TF-IDF", "detail": "baseline · n-grammes de 3 à 5 lettres",
     "toutes": {"p": 0.270, "hit": 50 / 63, "mrr": 0.708}, "meme": {"p": 0.296, "hit": 48 / 54, "mrr": 0.798}},
    {"cle": "embeddings", "nom": "e5-small", "detail": "multilingual-e5-small · 118 M · dim. 384",
     "toutes": {"p": 0.302, "hit": 50 / 63, "mrr": 0.702}, "meme": {"p": 0.278, "hit": 44 / 54, "mrr": 0.751}},
    {"cle": "labse", "nom": "LaBSE", "detail": "LaBSE · 472 M · dim. 768",
     "toutes": {"p": 0.286, "hit": 36 / 63, "mrr": 0.523}, "meme": {"p": 0.179, "hit": 29 / 54, "mrr": 0.429}},
    {"cle": "melange", "nom": "Mélange TF-IDF + e5", "detail": "α = 0,5 · réglé sur dev", "meilleur": True,
     "toutes": {"p": 0.296, "hit": 53 / 63, "mrr": 0.792}, "meme": {"p": 0.321, "hit": 51 / 54, "mrr": 0.890}},
    {"cle": "melange_labse", "nom": "Mélange TF-IDF + LaBSE", "detail": "α = 0,25 · réglé sur dev",
     "toutes": {"p": 0.349, "hit": 52 / 63, "mrr": 0.742}, "meme": {"p": 0.302, "hit": 48 / 54, "mrr": 0.794}},
]
REQUETES_TEST = {"toutes": 63, "meme": 54}

# Hit@3 selon la langue de la requête (scénario toutes_langues) ; 16 requêtes en anglais, 22 en français, 25 en somali.
PAR_LANGUE = {
    "colonnes": [("fr", "Français", 22), ("en", "English", 16), ("so", "Soomaali", 25)],
    "lignes": [
        {"nom": "TF-IDF", "valeurs": [0.727, 0.750, 0.880]},
        {"nom": "e5-small", "valeurs": [0.818, 0.938, 0.680]},
        {"nom": "LaBSE", "valeurs": [0.727, 0.500, 0.480]},
        {"nom": "Mélange TF-IDF + e5", "valeurs": [0.818, 0.812, 0.880]},
        {"nom": "Mélange TF-IDF + LaBSE", "valeurs": [0.818, 0.875, 0.800]},
    ],
}

# Requêtes dont la réponse n'existe que dans une autre langue : MRR sur 9 requêtes de test.
ENTRE_LANGUES = [
    {"nom": "TF-IDF", "hit": 0.22, "mrr": 0.17}, {"nom": "e5-small", "hit": 0.56, "mrr": 0.36},
    {"nom": "LaBSE", "hit": 0.44, "mrr": 0.46}, {"nom": "Mélange TF-IDF + e5", "hit": 0.22, "mrr": 0.20},
    {"nom": "Mélange TF-IDF + LaBSE", "hit": 0.44, "mrr": 0.43},
]

# Causes des cas difficiles (requêtes de test dont le premier résultat n'est pas un bon passage), lues une par une.
CAUSES = {
    "colonnes": ["TF-IDF", "e5-small", "LaBSE"],
    "lignes": [
        {"nom": "Faits voisins", "detail": "un passage proche par le sujet, pas par la réponse", "valeurs": [10, 17, 24]},
        {"nom": "Paraphrase", "detail": "la question n'a aucun mot en commun avec le passage", "valeurs": [7, 0, 0]},
        {"nom": "Autre langue", "detail": "la réponse n'existe que dans une autre langue", "valeurs": [7, 5, 0]},
        {"nom": "Passage attracteur", "detail": "un même passage général revient pour plusieurs questions", "valeurs": [0, 4, 14]},
        {"nom": "Annotation discutable", "detail": "le bon passage est peut-être mal annoté", "valeurs": [1, 0, 0]},
    ],
    "totaux": [25, 26, 38],
}

# Étude contrôlée du prétraitement pour TF-IDF : écart de MRR (requêtes dev) par rapport à la référence (0,770).
PRETRAITEMENT = {
    "reference": 0.770, "seuil": 0.05, "requetes_dev": 20,
    "variantes": [
        ("garder les accents", 0.001), ("garder la casse", 0.003), ("tf non sublinéaire", 0.000), ("min_df = 2", -0.025),
        ("max_df = 0,5", -0.001), ("caractères (2,4)", 0.002), ("caractères (4,6)", 0.000), ("analyseur char (sans bords de mots)", 0.010),
        ("mots vides ≥ 30 %", -0.023), ("mots vides ≥ 20 %", -0.027), ("mots vides ≥ 10 %", -0.106),
        ("mots vides ≥ 20 % + mots", -0.094), ("mots sans racines", -0.030), ("racines fr/en + mots", -0.031),
        ("racines fr/en + lettres", -0.023), ("lemmes spaCy + mots", -0.031), ("lemmes spaCy + lettres", -0.028),
        ("Jaccard (au lieu du cosinus)", -0.208),
    ],
}

# Fragmentation des mots par les tokenizers des modèles (sous-mots par mot).
FRAGMENTATION = [
    {"nom": "e5-small", "fr": 1.49, "en": 1.28, "so": 1.95}, {"nom": "LaBSE", "fr": 1.38, "en": 1.19, "so": 1.66},
]

# Réglage du poids α du mélange sur les requêtes dev : MRR (α = 0 : embeddings seuls ; α = 1 : TF-IDF seul).
ALPHA = {
    "e5-small": {"choisi": 0.5, "valeurs": [(0.0, 0.660), (0.25, 0.771), (0.5, 0.784), (0.75, 0.782), (1.0, 0.770)]},
    "LaBSE": {"choisi": 0.25, "valeurs": [(0.0, 0.576), (0.25, 0.781), (0.5, 0.777), (0.75, 0.775), (1.0, 0.770)]},
}
