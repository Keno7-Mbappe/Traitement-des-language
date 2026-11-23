# Projet 8 – Recherche sémantique dans des documents universitaires

TALN – Master 2 IAMD – Université de Djibouti – novembre 2026
Binôme : Kenedid Yacin Boulaleh et Abdoulrazack Houssein Mahamoud

## 1. Ce que fait le projet

Un étudiant pose une question, en **français, en anglais ou en somali**. Le système renvoie les **3 passages** de documents universitaires qui répondent le mieux. On compare cinq méthodes :

| Méthode | Principe |
|---|---|
| `tfidf` (baseline) | TF-IDF sur des n-grammes de caractères (3 à 5) et similarité cosinus |
| `embeddings` | vecteurs de phrases `intfloat/multilingual-e5-small` |
| `labse` | vecteurs de phrases `sentence-transformers/LaBSE` |
| `melange` | 0,5 × TF-IDF + 0,5 × e5-small (scores ramenés entre 0 et 1) |
| `melange_labse` | 0,25 × TF-IDF + 0,75 × LaBSE |

La qualité est mesurée avec **Precision@3, Hit@3 et MRR**, avec un jeu de requêtes séparé en réglage (`dev`) et test (`test`), puis une analyse de 89 cas difficiles.
Une mini-application web **Django** (extension demandée par l'énoncé) affiche le top 3.

### La chaîne de traitement, étape par étape

Le sujet demande toute la chaîne de traitement : *données → prétraitement → représentation → baseline → modèle → évaluation → analyse d'erreurs → démonstration*. Le notebook suit cet ordre (les numéros sont ceux de ses sections).

| Étape | Ce que fait le projet | Section |
|---|---|---|
| Données | corpus synthétique (14 documents en .txt et en PDF), fiche du corpus, annotation par « faits », requêtes séparées en dev et test, contrôle de fuite entre dev et test | 3, 5 |
| **Prétraitement** | nettoyage (`clean_text`) avec tests unitaires, mesures avant et après, six tokenizers comparés (split, regex, NLTK, spaCy), statistiques par langue, détection de la langue des requêtes, **étude contrôlée** : accents, mots vides, racines, lemmes, Jaccard | 4, 5, 6 |
| Représentation | TF-IDF par caractères ; embeddings e5-small et LaBSE ; tokenizers des modèles, évaluation intrinsèque et PCA | 6, 7 |
| Baseline | TF-IDF + cosinus, et plancher du hasard (équivalent d'un `DummyClassifier`) | 6 |
| Modèle | encodeurs pré-entraînés utilisés tels quels (sans entraînement), deux modèles et deux mélanges | 7 |
| Évaluation | Precision@3, Hit@3, MRR ; deux scénarios ; réglage sur dev seulement | 5, 8 |
| Analyse d'erreurs | 89 cas difficiles, cause de chacun, action proposée pour chaque cause | 9 |
| Démonstration | application Django (top 3), API JSON, fonction `rechercher` | 10 et application |

## 2. Contenu du dossier

```
Projet Final/
├── Projet8_Recherche_Semantique.ipynb   notebook : données, prétraitement, méthodes, résultats, analyse d'erreurs
├── Serveur_Embeddings_Colab.ipynb       (facultatif) serveur d'embeddings sur Colab pour la démonstration en ligne
├── data/
│   ├── documents/                       14 documents .txt (un passage = un bloc séparé par une ligne vide)
│   ├── documents_pdf/                   les mêmes 14 documents en PDF
│   ├── annotations.csv                  pour chaque passage, le ou les « faits » (informations) qu'il contient
│   └── requetes.csv                     83 requêtes : langue, type, fait attendu, partition dev ou test
├── data.zip                             copie compressée de data/ (à téléverser sur Colab)
├── application_django/                  mini-interface de recherche (voir partie 5)
│   ├── manage.py
│   ├── config/                          réglages Django
│   ├── recherche/                       moteur.py, views.py, gabarits, static/ (CSS, JavaScript, polices), tests.py, commande preparer_en_ligne
│   └── artefacts/                       fichiers produits par le notebook (TF-IDF, embeddings des passages, passages.csv)
├── requirements.txt                     dépendances du notebook et de l'application
├── requirements-web.txt                 dépendances de l'application seule (mise en ligne)
├── README.md
├── Projet8_Rapport_Recherche_Semantique.pdf   rapport (8 pages) ; la version Word modifiable est à côté
└── Soutenance_Projet8.pptx              diapositives
```

## 3. Les données

**Elles sont synthétiques** : l'énoncé autorise les données publiques, fournies, synthétiques ou anonymisées. Ce sont des documents universitaires **fictifs** (règlement des études, guide d'inscription, frais et bourses, syllabus du Master IAMD, calendrier, charte informatique, guide des stages, règlement de la bibliothèque, guide de l'étudiant international, guides en somali). Dates, montants et règles sont inventés. Ils ont été rédigés pour le projet **avec l'aide d'un assistant d'IA**.

| | Français | Anglais | Somali | Total |
|---|---|---|---|---|
| Documents | 7 | 2 | 5 | 14 |
| Passages | 123 | 42 | 67 | 232 |

- Un **passage** est une section d'un document (32 à 77 mots), avec son titre.
- Un **fait** est une information précise (par exemple `certificat_medical`). `annotations.csv` donne les faits de chaque passage : c'est ce qui permet de savoir quels passages sont de « bonnes réponses ». 131 faits différents, dont 65 n'existent que dans une seule langue (46 dans deux langues, 20 dans les trois) (utile pour tester la recherche entre langues).
- Les **83 requêtes** (28 en français, 22 en anglais, 33 en somali) sont de trois types : `mots_communs`, `paraphrase`, `court`. 20 servent à régler les paramètres (`dev`) et 63 au résultat final (`test`).
- **Le somali a été écrit avec l'aide d'une IA et n'a pas encore été relu par un locuteur.**

## 4. Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

L'installation a été vérifiée dans un environnement vierge (Python 3.13, Django 6.1, pandas 3.1, numpy 2.5, scikit-learn 1.9) : `python manage.py test` réussit tous les tests.

### Exécuter le notebook (méthode recommandée : Google Colab)

1. Ouvrir `Projet8_Recherche_Semantique.ipynb` dans Google Colab.
2. Tout exécuter. La première cellule installe les bibliothèques et demande de **téléverser `data.zip`**.
3. À la fin, `artefacts.zip` se télécharge : il contient les fichiers pour Django. Les décompresser dans `application_django/artefacts/`.

Sur un ordinateur local (installer en plus `jupyter`), le notebook tourne avec **TF-IDF seul** : les modèles d'embeddings demandent beaucoup de mémoire et sont activés automatiquement sur Colab.

## 5. Lancer l'application Django

```
cd application_django
python manage.py runserver
```

Puis ouvrir http://127.0.0.1:8000 (si ce port est pris : `python manage.py runserver 8001`). L'application s'appelle **Scriptorium** et compte quatre pages :

- **Recherche** : formulaire, découpage en jetons en direct, choix de la méthode, exemples (qui marchent et cas difficiles). Chaque résultat affiche son titre, sa langue, son document, son score, les **mots qu'il partage avec la question** (surlignés) et sa position sur la **carte des passages** (projection PCA des vecteurs e5-small ; la question y apparaît en étoile avec les méthodes e5).
- **Méthode** : la chaîne de traitement en six étapes, les cinq méthodes, un démonstrateur de Precision@3, Hit@3 et MRR, et un exemple réel de réponse de l'API.
- **Résultats** : les mesures du notebook (comparaison des méthodes, par langue, entre langues, causes d'erreur, étude du prétraitement).
- **Corpus** : les 14 documents et leurs sections, calculés sur `passages.csv`.

Le site existe en thème clair (beige) et sombre (espresso), choisi par le système ou par le bouton de l'en-tête ; `?theme=clair` ou `?theme=sombre` dans l'adresse force un thème. Les polices sont incluses dans le dossier (`static/recherche/fonts/`, licence SIL OFL) : l'interface s'affiche identique sans connexion.

- **Sans réglage**, seule la méthode **TF-IDF** est disponible.
- **Pour utiliser les méthodes à embeddings en ligne** (internet nécessaire) : créer un jeton gratuit sur huggingface.co (*Settings > Access Tokens*, avec le droit d'appeler les *Inference Providers*), puis, **avant** de lancer l'application :

  ```
  $env:HF_TOKEN = "votre_jeton"          (PowerShell)        set HF_TOKEN=votre_jeton          (cmd)
  python manage.py preparer_en_ligne
  python manage.py runserver
  ```

  `preparer_en_ligne` vérifie que le service en ligne renvoie les mêmes vecteurs que le notebook, puis garde en cache les vecteurs des questions d'exemple (elles marcheront même sans connexion). Avec `--evaluation`, il fait de même pour les 83 requêtes du jeu d'évaluation. **Rien n'est mis en cache si le service n'est pas fidèle au notebook** (cosinus inférieur à 0,99 sur les passages de contrôle).
- Le jeton n'est lu que dans la variable d'environnement, il n'est ni affiché ni enregistré.
- **Autre solution, sans Hugging Face : un serveur d'embeddings sur Colab.** Le notebook `Serveur_Embeddings_Colab.ipynb` fait tourner les deux mêmes modèles (e5-small et LaBSE) sur Colab et les rend joignables par une adresse temporaire (tunnel Cloudflare, sans compte). Aucun jeton n'est nécessaire.
  1. Ouvrir `Serveur_Embeddings_Colab.ipynb` dans Colab et exécuter les cellules dans l'ordre (chargement des modèles : 1 à 3 minutes).
  2. La cellule 3 se teste elle-même, puis affiche l'adresse du serveur, une clé secrète et les commandes à copier :

     ```
     $env:PROJET8_SERVEUR = "https://....trycloudflare.com"          (PowerShell)        set PROJET8_SERVEUR=...          (cmd)
     $env:PROJET8_CLE = "la clé affichée par Colab"                  (PowerShell)        set PROJET8_CLE=...              (cmd)
     python manage.py preparer_en_ligne
     python manage.py runserver
     ```
  3. Laisser la cellule Colab active et l'onglet ouvert pendant la démonstration. L'adresse et la clé changent à chaque exécution de Colab.

  Si `PROJET8_SERVEUR` est défini, l'application l'utilise avant Hugging Face (`PROJET8_MODE=en_ligne` force Hugging Face, `PROJET8_MODE=local` coupe les deux). `preparer_en_ligne` fonctionne avec l'un ou l'autre service.
- **Confidentialité** : en mode en ligne, le **texte de la question** (jamais les documents) est envoyé au service qui calcule son vecteur : Hugging Face, ou le serveur Colab (il passe alors par Cloudflare et Google Colab). La clé du serveur empêche un inconnu qui trouverait l'adresse de s'en servir : ne la publiez pas dans une capture d'écran.
- Si le port 8000 est déjà utilisé sur l'ordinateur, lancer `python manage.py runserver 8001` et ouvrir http://127.0.0.1:8001.
- Si une méthode n'est pas disponible (pas de jeton ni de serveur, pas de connexion, erreur du service, cellule Colab arrêtée), l'application le dit et affiche les résultats TF-IDF.
- **API JSON** : `http://127.0.0.1:8000/api/recherche/?q=bibliotheque%20horaires&methode=tfidf` (mêmes champs que la fonction `reponse_api` du notebook).
- **Tests** : `python manage.py test` (104 tests, sans réseau, sans jeton, sans Colab : les services en ligne sont simulés, le vrai code du serveur Colab est testé avec de faux modèles, et les pages, le surlignage, la carte et la cohérence des résultats avec ce README sont contrôlés).

### Questions prêtes (fonctionnent sans service)

Le dépôt contient `artefacts/cache_vecteurs_requetes.json` (vecteurs e5-small et LaBSE de 83 questions, calculés par le serveur Colab après vérification) et `artefacts/questions_pretes.json` (leur liste). Ces questions marchent avec **toutes** les méthodes, même quand aucun service d'embeddings n'est branché, par exemple sur le site en ligne : la page de recherche les liste. Recalculées uniquement à partir de ce cache, les cinq méthodes redonnent exactement les résultats du notebook (Hit@3 et MRR identiques sur les 63 requêtes de test) ; un test le vérifie. Toute autre question retombe sur TF-IDF, ou utilise le service s'il est branché.

### Mise en ligne (Render)

**Application en ligne : https://recherche-document-universitaire.onrender.com** (dépôt : https://github.com/Keno7-Mbappe/Traitement-des-language). Elle tourne sur Render (service web gratuit) au lieu de l'ordinateur ; chaque publication sur la branche `main` la met à jour toute seule. Le dépôt GitHub contient tout ce qu'il faut : `requirements-web.txt` (dépendances de l'application seule), `application_django/` et ses `artefacts/`. Réglages du service :

| Réglage | Valeur |
|---|---|
| Type, langage | Web Service, Python 3 |
| Build Command | `pip install -r requirements-web.txt && cd application_django && python manage.py collectstatic --noinput` |
| Start Command | `gunicorn config.wsgi:application --chdir application_django --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` |
| Health Check Path | `/` |
| `PYTHON_VERSION` | `3.13.5` |
| `DJANGO_SECRET_KEY` | une longue valeur aléatoire (obligatoire : l'application refuse de démarrer sans elle en ligne) |
| `HF_TOKEN` ou `PROJET8_SERVEUR` + `PROJET8_CLE` | facultatif : active les méthodes à embeddings (Hugging Face, ou serveur Colab) ; sans eux, TF-IDF fonctionne seul |

Remarques : sur l'offre gratuite, le service s'endort après environ 15 minutes sans visite et met une minute à se réveiller (ouvrir le site avant la démonstration). Le disque est temporaire : le cache des questions disparaît à chaque redémarrage. En ligne, la variable `RENDER` (définie par la plateforme) coupe le mode développement, active WhiteNoise pour les fichiers statiques et accepte l'adresse du site ; les secrets se saisissent dans le tableau de bord de Render, jamais dans le dépôt.

## 6. Résultats (63 requêtes de test, K = 3)

| Méthode | toutes langues : Hit@3 | MRR | même langue : Hit@3 | MRR |
|---|---|---|---|---|
| TF-IDF (baseline) | 0,79 | 0,71 | 0,89 | 0,80 |
| Embeddings e5-small | 0,79 | 0,70 | 0,81 | 0,75 |
| Embeddings LaBSE | 0,57 | 0,52 | 0,54 | 0,43 |
| **Mélange TF-IDF + e5-small** | **0,84** | **0,79** | **0,94** | **0,89** |
| Mélange TF-IDF + LaBSE | 0,83 | 0,74 | 0,89 | 0,79 |

« Toutes langues » : sont bons tous les passages qui expriment le fait attendu, quelle que soit leur langue. « Même langue » : seulement ceux écrits dans la langue de la requête. La mesure se fait par passage (comme l'extension demandée, top 3 des passages), ce qui est plus strict que par document : un bon passage trouvé entraîne son document.

La Precision@3 (part des 3 passages affichés qui sont bons) :

| Méthode | toutes langues : P@3 | même langue : P@3 |
|---|---|---|
| TF-IDF (baseline) | 0,27 | 0,30 |
| Embeddings e5-small | 0,30 | 0,28 |
| Embeddings LaBSE | 0,29 | 0,18 |
| Mélange TF-IDF + e5-small | 0,30 | **0,32** |
| Mélange TF-IDF + LaBSE | **0,35** | 0,30 |

Comment lire ces résultats : le mélange TF-IDF + e5-small a le meilleur MRR (0,79 contre 0,74 pour le mélange LaBSE et 0,71 pour la baseline). En Hit@3, il gagne 3 requêtes sur la baseline mais une seule sur le mélange LaBSE (écart de 0,016, sous notre seuil de 0,03 : une requête change Hit@3 de 0,016). En Precision@3 « toutes langues », le mélange LaBSE est premier : il place en moyenne 1,27 bon passage dans le top 3 par requête réussie, contre 1,06, car cette métrique récompense le fait de ramener plusieurs versions du même fait dans plusieurs langues. Le choix entre les deux mélanges dépend donc de la métrique ; en « même langue », le mélange TF-IDF + e5-small est premier dans les trois.
Le détail (par langue, par type, requêtes croisées, 89 cas difficiles avec leur cause) est dans le notebook et dans le rapport.

**Ce que le prétraitement a montré** (notebook, sections 4 à 7) :

- Le nettoyage ne change que les sauts de ligne : aucun mot supprimé, aucun doublon, aucune URL, aucun e-mail. La fonction complète donne le même texte que celle du pipeline.
- L'apostrophe n'a pas le même rôle selon la langue (élision en français, lettre du mot en somali) : aucun tokenizer testé ne convient aux trois langues, et spaCy n'a pas de langue somali.
- La langue d'une requête se détecte à 0,976 d'accuracy (TF-IDF caractères + Naive Bayes, contre 0,337 pour la classe majoritaire).
- **Aucune des 18 variantes testées** (accents, casse, mots vides, racines, lemmes, bornes de fréquence, Jaccard) n'améliore le MRR sur dev d'au moins 0,05 : le pipeline TF-IDF est donc gardé tel quel. Les mêmes 5 requêtes dev échouent avec 16 de ces variantes sur 18 : leur problème est le vocabulaire, pas le prétraitement.
- Le hasard donne Hit@3 ≈ 0,03.
- Le somali est le plus fragmenté par les tokenizers des deux modèles (1,95 sous-mot par mot pour e5-small contre 1,49 en français). L'espace de e5-small se range **par langue** (96 % des voisins sont dans la même langue), celui de LaBSE **par sens** : cela va dans le sens de LaBSE meilleur entre langues (MRR, sur 9 requêtes seulement : une tendance, pas une preuve) et d'e5-small dans la langue de la question.

## 7. Limites

- Corpus **synthétique et petit** (232 passages) : les scores valent pour ce corpus, pas pour de vrais documents.
- **Somali** non relu par un locuteur ; les embeddings y sont nettement moins bons que TF-IDF.
- Peu de requêtes (63 en test) : une requête change Hit@3 de 0,016.
- Requêtes écrites par nous : elles peuvent être plus proches des passages que de vraies questions.
- Une seule bonne réponse par requête : beaucoup d'erreurs sont des « faits voisins » défendables.
- L'étude du prétraitement repose sur 20 requêtes dev : un petit effet ne serait pas visible. Le somali n'a ni racineur ni lemmatiseur, et ses mots vides (tirés du corpus) n'ont pas été validés par un locuteur.
- 13 requêtes de test visent un fait aussi visé par une requête dev : le test n'est pas parfaitement indépendant du réglage.
- Le corpus synthétique est propre (ni HTML, ni URL, ni e-mail) : le nettoyage n'a été validé que sur des cas de test, pas sur de vrais PDF.

## 8. Modèles utilisés (documentation)

| Modèle | Langues | Paramètres | Dimension | Longueur max | Limites |
|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | environ 100 (celles de XLM-R) | 118 millions | 384 | 512 tokens | couverture du somali non garantie ; demande les préfixes `query: ` et `passage: ` |
| `sentence-transformers/LaBSE` | 109 | 472 millions | 768 | 256 tokens | entraîné sur des phrases traduites, pas sur des paires question et passage |

Paramètres, dimensions et longueurs sont ceux affichés par le notebook. Aucun passage n'est coupé (le plus long fait 133 tokens avec e5-small et 114 avec LaBSE).

## 9. Références

- Cours TALN M2 IAMD (Dr H. A. Assowe), Université de Djibouti, 2026 : séance 1 (introduction, chaîne de traitement, tokenisation), séance 2 (prétraitement : nettoyage, tokenisation, mots vides, racines et lemmes, contrôle qualité), séance 3 (représentation vectorielle : TF-IDF, cosinus, Jaccard), séance 4 (analyse linguistique, lemmatisation), séance 5 et TP5 (classification : baseline, métriques, analyse d'erreurs), séance 6 (représentations distribuées), séance 8 (modèles pré-entraînés, tokenizers), séances 9-10 (évaluation, protocole expérimental), TD 11-12 (prototype reproductible, analyse d'erreurs). La fonction de nettoyage `clean_text` et la regex de tokenisation en sont reprises, puis adaptées.
- Manning, Raghavan, Schütze, *Introduction to Information Retrieval*, 2008.
- Reimers, Gurevych, *Sentence-BERT*, EMNLP 2019.
- Wang et al., *Multilingual E5 Text Embeddings: A Technical Report*, 2024.
- Feng et al., *Language-agnostic BERT Sentence Embedding*, ACL 2022.
- Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale*, ACL 2020.
