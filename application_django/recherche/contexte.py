"""Variables disponibles dans tous les gabarits (pied de page : où part le texte de la question)."""
from . import moteur


def contexte_global(request):
    return {"service_en_ligne": moteur.service_en_ligne()}
