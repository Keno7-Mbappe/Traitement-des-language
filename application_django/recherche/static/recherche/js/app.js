/* Scriptorium : comportements communs à toutes les pages (thème, menu, apparitions, compteurs)
   et à la page de recherche (jetons en direct, calcul en cours, copie). Aucune bibliothèque, aucune connexion. */
(() => {
  "use strict";
  const racine = document.documentElement;
  const reduit = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const formatFr = (n, d = 0) => n.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });

  /* ---------- notifications ---------- */
  function notifier(message) {
    const zone = document.querySelector(".toasts");
    if (!zone) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    zone.appendChild(toast);
    setTimeout(() => toast.remove(), 2700);
  }

  /* ---------- découpage en jetons : même regex que le notebook (apostrophes ' et ’, trait d'union, nombres) ---------- */
  const REGEX_JETON = /[\p{L}\p{M}]+(?:['’-][\p{L}\p{M}]+)*|\d+(?:[.,]\d+)?/gu;
  const jetons = (texte) => texte.toLowerCase().match(REGEX_JETON) || [];

  /* Affiche la liste de jetons en réutilisant les éléments déjà là : seuls les nouveaux jetons s'animent. */
  function afficherJetons(liste, texte, maximum = 30) {
    const mots = jetons(texte);
    const visibles = mots.slice(0, maximum);
    while (liste.children.length > visibles.length) liste.lastElementChild.remove();
    visibles.forEach((mot, i) => {
      const existant = liste.children[i];
      if (existant && existant.textContent === mot) return;
      const li = document.createElement("li");
      li.className = "jeton";
      li.style.setProperty("--k", Math.min(i, 12));
      li.textContent = mot;
      if (existant) existant.replaceWith(li); else liste.appendChild(li);
    });
    if (mots.length > maximum) {
      const plus = document.createElement("li");
      plus.className = "jeton";
      plus.textContent = `+ ${mots.length - maximum}`;
      liste.appendChild(plus);
    }
    return mots.length;
  }
  window.Scriptorium = { afficherJetons, jetons, notifier, formatFr, reduit };

  /* ---------- thème clair / sombre ---------- */
  const sombreSysteme = window.matchMedia("(prefers-color-scheme: dark)");
  if (racine.dataset.theme === "auto") racine.dataset.theme = sombreSysteme.matches ? "dark" : "light";   // état explicite, sans l'enregistrer
  document.querySelectorAll("[data-theme-toggle]").forEach((bouton) => {
    bouton.addEventListener("click", () => {
      const suivant = racine.dataset.theme === "dark" ? "light" : "dark";
      racine.dataset.theme = suivant;
      try { localStorage.setItem("scriptorium-theme", suivant); } catch (e) { /* stockage indisponible : le choix vaut pour cette page */ }
    });
  });

  /* ---------- en-tête : ombre au défilement et menu mobile ---------- */
  const entete = document.querySelector("[data-entete]");
  if (entete) {
    let attente = false;
    const surDefilement = () => {
      attente = false;
      entete.classList.toggle("est-defile", window.scrollY > 8);
    };
    window.addEventListener("scroll", () => { if (!attente) { attente = true; requestAnimationFrame(surDefilement); } }, { passive: true });
    surDefilement();

    const burger = entete.querySelector("[data-menu]");
    const fermer = () => {
      entete.classList.remove("est-ouvert");
      if (burger) { burger.setAttribute("aria-expanded", "false"); burger.setAttribute("aria-label", "Ouvrir le menu"); }
    };
    if (burger) {
      burger.addEventListener("click", () => {
        const ouvert = entete.classList.toggle("est-ouvert");
        burger.setAttribute("aria-expanded", String(ouvert));
        burger.setAttribute("aria-label", ouvert ? "Fermer le menu" : "Ouvrir le menu");
      });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape" && entete.classList.contains("est-ouvert")) { fermer(); burger.focus(); } });
      window.addEventListener("resize", () => { if (window.innerWidth >= 900) fermer(); });
      entete.querySelectorAll(".nav__lien").forEach((lien) => lien.addEventListener("click", fermer));
    }
  }

  /* ---------- apparition au défilement et compteurs ---------- */
  function compter(element) {
    const cible = parseFloat(element.dataset.compter);
    if (Number.isNaN(cible)) return;
    const decimales = parseInt(element.dataset.decimales || "0", 10);
    if (reduit) { element.textContent = formatFr(cible, decimales); return; }
    const duree = 1200;
    const debut = performance.now();
    const pas = (maintenant) => {
      const t = Math.min(1, (maintenant - debut) / duree);
      const eased = 1 - Math.pow(1 - t, 4);
      element.textContent = formatFr(cible * eased, decimales);
      if (t < 1) requestAnimationFrame(pas);
    };
    requestAnimationFrame(pas);
  }
  const observeur = "IntersectionObserver" in window ? new IntersectionObserver((entrees) => {
    entrees.forEach((entree) => {
      if (!entree.isIntersecting) return;
      const cible = entree.target;
      if (cible.matches("[data-reveler]")) cible.classList.add("est-visible");
      if (cible.matches("[data-compter]")) compter(cible);
      observeur.unobserve(cible);
    });
  }, { threshold: 0.18, rootMargin: "0px 0px -6% 0px" }) : null;
  document.querySelectorAll("[data-reveler], [data-compter]").forEach((element) => {
    if (observeur) observeur.observe(element);
    else { element.classList.add("est-visible"); }
  });

  /* ---------- page de recherche ---------- */
  const composeur = document.querySelector("[data-composeur]");
  if (composeur) {
    const champ = composeur.querySelector("[data-champ]");
    const effacer = composeur.querySelector("[data-effacer]");
    const zoneJetons = composeur.querySelector("[data-jetons]");
    const listeJetons = composeur.querySelector("[data-jetons-liste]");

    const mettreAJour = () => {
      const texte = champ.value;
      effacer.hidden = texte.length === 0;
      const nombre = afficherJetons(listeJetons, texte);
      zoneJetons.hidden = nombre === 0;
    };
    champ.addEventListener("input", mettreAJour);
    mettreAJour();
    effacer.addEventListener("click", () => { champ.value = ""; mettreAJour(); champ.focus(); });

    document.addEventListener("keydown", (e) => {
      const ecrit = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName);
      if (e.key === "/" && !ecrit && !e.ctrlKey && !e.metaKey && !e.altKey) { e.preventDefault(); champ.focus(); champ.select(); }
    });

    const enCalcul = (actif) => { document.body.classList.toggle("est-en-calcul", actif); composeur.classList.toggle("est-en-calcul", actif); };
    composeur.addEventListener("submit", () => {
      if (!champ.value.trim()) return;
      enCalcul(true);
    });
    window.addEventListener("pageshow", () => enCalcul(false));                    // retour arrière : on ne reste pas « en calcul »

    const zone = document.querySelector("[data-zone-resultats]");
    if (zone) {
      if (zone.getBoundingClientRect().top > window.innerHeight * 0.75) {          // résultats hors de vue (mobile) : on les amène à l'écran
        setTimeout(() => zone.scrollIntoView({ behavior: reduit ? "auto" : "smooth", block: "start" }), 250);
      }
    } else if (window.innerWidth >= 900 && !window.matchMedia("(pointer: coarse)").matches && !champ.value) {
      champ.focus({ preventScroll: true });
    }
  }

  /* ---------- résultats : copie d'un passage, lien avec la carte ---------- */
  async function copierTexte(texte) {
    try { await navigator.clipboard.writeText(texte); return true; } catch (e) { /* repli ci-dessous */ }
    const zone = document.createElement("textarea");
    zone.value = texte; zone.setAttribute("readonly", ""); zone.style.cssText = "position:fixed;opacity:0";
    document.body.appendChild(zone); zone.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    zone.remove();
    return ok;
  }
  document.querySelectorAll("[data-copier]").forEach((bouton) => {
    bouton.addEventListener("click", async () => {
      const carte = bouton.closest("[data-resultat]");
      const texte = `${carte.querySelector(".resultat__titre").textContent}\n${carte.querySelector(".resultat__texte").textContent}`;
      const ok = await copierTexte(texte);
      notifier(ok ? "Passage copié" : "Copie impossible : sélectionnez le texte à la main");
      if (ok) {
        const libelle = bouton.querySelector("span");
        const avant = libelle.textContent;
        bouton.classList.add("est-copie");
        libelle.textContent = "Copié";
        setTimeout(() => { bouton.classList.remove("est-copie"); libelle.textContent = avant; }, 1800);
      }
    });
  });
  document.querySelectorAll("[data-resultat]").forEach((carte) => {
    const signaler = (id) => document.dispatchEvent(new CustomEvent("resultat:survol", { detail: { id } }));
    carte.addEventListener("mouseenter", () => signaler(carte.dataset.id));
    carte.addEventListener("mouseleave", () => signaler(null));
    carte.addEventListener("focusin", () => signaler(carte.dataset.id));
    carte.addEventListener("focusout", () => signaler(null));
  });
  document.addEventListener("carte:survol", (e) => {
    document.querySelectorAll("[data-resultat]").forEach((carte) => carte.classList.toggle("est-actif", carte.dataset.id === e.detail.id));
  });
})();
