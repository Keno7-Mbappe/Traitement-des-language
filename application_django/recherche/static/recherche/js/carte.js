/* Carte des passages : projection PCA (2 axes) des vecteurs e5-small, un point par passage.
   Les points s'installent, flottent doucement, révèlent le titre au survol ; les 3 résultats sont cerclés et numérotés,
   et la question, quand elle a un vecteur e5, est une étoile reliée à ses résultats. Canvas 2D, sans bibliothèque. */
(() => {
  "use strict";
  const source = document.getElementById("carte-donnees");
  const panneau = document.querySelector("[data-carte-panneau]");
  if (!source || !panneau) return;
  const canvas = panneau.querySelector("canvas[data-carte]");
  const bulle = panneau.querySelector("[data-bulle]");
  const toile = panneau.querySelector(".carte__toile");
  const ctx = canvas.getContext("2d");
  const donnees = JSON.parse(source.textContent);
  const reduit = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NOMS = { fr: "Français", en: "English", so: "Soomaali" };

  const points = donnees.points.map((p, i) => ({ ...p, phase: i * 2.399, rang: 0 }));
  const parId = new Map(points.map((p) => [p.id, p]));
  (donnees.resultats || []).forEach((id, i) => { const p = parId.get(id); if (p) p.rang = i + 1; });
  const resultats = points.filter((p) => p.rang).sort((a, b) => a.rang - b.rang);
  const question = donnees.question || null;
  const actives = new Set(["fr", "en", "so"]);
  let largeur = 0, hauteur = 0, dpr = 1, couleurs = {}, debut = performance.now(), survol = null, mis_en_avant = null, visible = true, rafId = 0;

  function lireCouleurs() {
    const s = getComputedStyle(document.documentElement);
    const v = (nom) => s.getPropertyValue(nom).trim();
    couleurs = { fr: v("--fr-pt"), en: v("--en-pt"), so: v("--so-pt"), encre: v("--ink"), fond: v("--surface"), accent: v("--accent"), or: v("--gold"), marque: v("--brand"), ligne: v("--line-2"), onBrand: v("--on-brand") };
  }

  function dimensionner() {
    const rect = toile.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    largeur = Math.max(1, Math.round(rect.width));
    hauteur = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(largeur * dpr);
    canvas.height = Math.round(hauteur * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  const MARGE = 30, LIMITE = 1.22;
  const DECALAGES = { 1: [15, -15], 2: [16, 15], 3: [-16, -15] };
  const enX = (x) => MARGE + ((x + LIMITE) / (2 * LIMITE)) * (largeur - 2 * MARGE);
  const enY = (y) => MARGE + (1 - (y + LIMITE) / (2 * LIMITE)) * (hauteur - 2 * MARGE);

  function position(p, t) {
    const flotte = reduit ? 0 : 1.7;
    return [enX(p.x) + Math.sin(t * 0.0008 + p.phase) * flotte, enY(p.y) + Math.cos(t * 0.00065 + p.phase * 1.3) * flotte];
  }

  function etoile(cx, cy, rayon) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? rayon : rayon * 0.45;
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      ctx[i ? "lineTo" : "moveTo"](cx + Math.cos(a) * r, cy + Math.sin(a) * r);
    }
    ctx.closePath();
  }

  function dessiner(maintenant) {
    const t = maintenant - debut;
    ctx.clearRect(0, 0, largeur, hauteur);

    // repères discrets : un espace, pas un nuage
    ctx.save();
    ctx.strokeStyle = couleurs.ligne; ctx.globalAlpha = 0.28; ctx.lineWidth = 1; ctx.setLineDash([2, 6]);
    for (const f of [0.4, 0.8, 1.2]) {
      ctx.beginPath(); ctx.ellipse(enX(0), enY(0), (f / (2 * LIMITE)) * (largeur - 2 * MARGE), (f / (2 * LIMITE)) * (hauteur - 2 * MARGE), 0, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();

    const placement = new Map();
    // points
    points.forEach((p, i) => {
      const [x, y] = position(p, t);
      placement.set(p, [x, y]);
      const apparition = reduit ? 1 : Math.min(1, Math.max(0, (t - 120 - i * 3.2) / 420));
      if (apparition <= 0) return;
      const actif = actives.has(p.l);
      const estSurvol = p === survol || p === mis_en_avant;
      const rayon = (estSurvol ? 6.5 : p.rang ? 5 : 3.4) * (0.4 + 0.6 * apparition);
      ctx.globalAlpha = (actif ? (resultats.length && !p.rang ? 0.5 : 0.88) : 0.1) * apparition;
      ctx.fillStyle = couleurs[p.l];
      ctx.beginPath(); ctx.arc(x, y, rayon, 0, Math.PI * 2); ctx.fill();
      if (estSurvol) { ctx.globalAlpha = 0.9; ctx.lineWidth = 2; ctx.strokeStyle = couleurs.encre; ctx.beginPath(); ctx.arc(x, y, rayon + 4, 0, Math.PI * 2); ctx.stroke(); }
    });
    ctx.globalAlpha = 1;

    // lien question -> résultats (trait animé), puis anneaux numérotés
    if (question && resultats.length) {
      const [qx, qy] = [enX(question[0]), enY(question[1])];
      ctx.save();
      ctx.strokeStyle = couleurs.accent; ctx.lineWidth = 1.6; ctx.setLineDash([5, 5]); ctx.lineDashOffset = reduit ? 0 : -t * 0.02; ctx.globalAlpha = 0.85;
      resultats.forEach((p) => { const [x, y] = placement.get(p); ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(x, y); ctx.stroke(); });
      ctx.restore();
    }
    resultats.forEach((p) => {
      const [x, y] = placement.get(p);
      const pulse = reduit ? 0 : (Math.sin(t * 0.004 + p.rang) + 1) * 1.2;
      ctx.strokeStyle = couleurs.accent; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.arc(x, y, 11 + pulse, 0, Math.PI * 2); ctx.stroke();
      const [dx, dy] = DECALAGES[p.rang] || DECALAGES[1];          // les trois numéros partent dans des directions différentes
      ctx.fillStyle = couleurs.marque; ctx.beginPath(); ctx.arc(x + dx, y + dy, 9, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = couleurs.onBrand; ctx.font = "600 11px 'JetBrains Mono', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(p.rang), x + dx, y + dy + 0.5);
    });
    if (question) {
      const [qx, qy] = [enX(question[0]), enY(question[1])];
      const halo = reduit ? 0 : (Math.sin(t * 0.003) + 1) * 4;
      ctx.fillStyle = couleurs.or; ctx.globalAlpha = 0.28; ctx.beginPath(); ctx.arc(qx, qy, 14 + halo, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
      ctx.fillStyle = couleurs.or; ctx.strokeStyle = couleurs.marque; ctx.lineWidth = 1.6; etoile(qx, qy, 11); ctx.fill(); ctx.stroke();
    }
    return placement;
  }

  let dernierPlacement = new Map();
  function boucle(maintenant) {
    dernierPlacement = dessiner(maintenant);
    rafId = visible && !document.hidden && !reduit ? requestAnimationFrame(boucle) : 0;
  }
  function relancer() {
    if (reduit) { dernierPlacement = dessiner(performance.now() + 5000); return; }
    if (!rafId && visible && !document.hidden) rafId = requestAnimationFrame(boucle);
  }

  /* ---------- interactions ---------- */
  function pointProche(px, py) {
    let meilleur = null, distance = 14;
    for (const p of points) {
      if (!actives.has(p.l)) continue;
      const place = dernierPlacement.get(p);
      if (!place) continue;
      const d = Math.hypot(place[0] - px, place[1] - py);
      if (d < distance) { distance = d; meilleur = p; }
    }
    return meilleur;
  }
  function afficherBulle(p) {
    if (!p) { bulle.hidden = true; return; }
    const [x, y] = dernierPlacement.get(p);
    bulle.replaceChildren();
    const titre = document.createElement("b"); titre.textContent = p.t;
    const detail = document.createElement("span"); detail.textContent = `${NOMS[p.l]} · ${p.d}`;
    bulle.append(titre, detail);
    bulle.hidden = false;
    const demi = bulle.offsetWidth / 2;
    bulle.style.left = `${Math.min(largeur - demi - 6, Math.max(demi + 6, x))}px`;
    bulle.style.top = `${Math.max(y, bulle.offsetHeight + 24)}px`;
  }
  function surPointeur(e) {
    const rect = canvas.getBoundingClientRect();
    const p = pointProche(e.clientX - rect.left, e.clientY - rect.top);
    if (p === survol) return;
    survol = p;
    afficherBulle(p);
    canvas.style.cursor = p ? "pointer" : "default";
    document.dispatchEvent(new CustomEvent("carte:survol", { detail: { id: p && p.rang ? p.id : null } }));
    if (reduit) relancer();
  }
  canvas.addEventListener("pointermove", surPointeur);
  canvas.addEventListener("pointerdown", surPointeur);
  canvas.addEventListener("pointerleave", () => { survol = null; afficherBulle(null); document.dispatchEvent(new CustomEvent("carte:survol", { detail: { id: null } })); if (reduit) relancer(); });
  document.addEventListener("resultat:survol", (e) => { mis_en_avant = e.detail.id ? parId.get(e.detail.id) : null; if (reduit) relancer(); });

  panneau.querySelectorAll("[data-langue]").forEach((bouton) => {
    bouton.addEventListener("click", () => {
      const langue = bouton.dataset.langue;
      const actif = actives.has(langue);
      if (actif && actives.size === 1) return;                         // au moins une langue reste visible
      if (actif) actives.delete(langue); else actives.add(langue);
      bouton.setAttribute("aria-pressed", String(!actif));
      if (reduit) relancer();
    });
  });

  /* ---------- démarrage ---------- */
  lireCouleurs();
  dimensionner();
  new ResizeObserver(() => { dimensionner(); relancer(); if (reduit) relancer(); }).observe(toile);
  new MutationObserver(() => { lireCouleurs(); if (reduit) relancer(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entrees) => { visible = entrees[0].isIntersecting; if (visible) relancer(); }).observe(canvas);
  }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) relancer(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => reduit && relancer());
  relancer();
})();
