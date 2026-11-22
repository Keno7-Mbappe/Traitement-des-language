/* Scriptorium : composants des pages Méthode, Résultats et Corpus (onglets, découpage en direct, métriques, comparateur, filtre). */
(() => {
  "use strict";
  const { afficherJetons, formatFr } = window.Scriptorium;

  /* ---------- onglets des étapes (page Méthode) ---------- */
  document.querySelectorAll("[data-onglets]").forEach((groupe) => {
    const onglets = [...groupe.querySelectorAll("[data-onglet]")];
    const panneaux = [...groupe.querySelectorAll("[data-panneau]")];
    const choisir = (indice, focus = false) => {
      onglets.forEach((onglet, i) => {
        const actif = i === indice;
        onglet.setAttribute("aria-selected", String(actif));
        onglet.tabIndex = actif ? 0 : -1;
        panneaux[i].hidden = !actif;
        if (actif && focus) onglet.focus();
        if (actif) onglet.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
      });
    };
    const depuisAdresse = () => {
      const trouve = /^#etape-([1-9])$/.exec(location.hash);
      if (trouve && onglets[trouve[1] - 1]) choisir(trouve[1] - 1);
    };
    depuisAdresse();
    window.addEventListener("hashchange", depuisAdresse);
    onglets.forEach((onglet, i) => {
      onglet.addEventListener("click", () => { choisir(i); history.replaceState(null, "", `#etape-${i + 1}`); });
      onglet.addEventListener("keydown", (e) => {
        const dernier = onglets.length - 1;
        const touches = { ArrowRight: (i + 1) % onglets.length, ArrowLeft: (i + dernier) % onglets.length, Home: 0, End: dernier };
        if (e.key in touches) { e.preventDefault(); choisir(touches[e.key], true); }
      });
    });
  });

  /* ---------- découpage en jetons en direct ---------- */
  document.querySelectorAll("[data-demo-texte]").forEach((champ) => {
    const liste = document.querySelector("[data-demo-jetons]");
    const maj = () => afficherJetons(liste, champ.value, 40);
    champ.addEventListener("input", maj);
    maj();
  });

  /* ---------- démonstrateur de métriques ---------- */
  document.querySelectorAll("[data-metriques]").forEach((bloc) => {
    const curseur = bloc.querySelector("[data-rang]");
    const sortieRang = bloc.querySelector("[data-sortie-rang]");
    const liste = bloc.querySelector("[data-liste]");
    const radios = [...bloc.querySelectorAll("input[name='nombre-bons']")];
    const champs = { p: bloc.querySelector("[data-sortie-p]"), hit: bloc.querySelector("[data-sortie-hit]"), rr: bloc.querySelector("[data-sortie-rr]") };
    const textes = { p: bloc.querySelector("[data-texte-p]"), hit: bloc.querySelector("[data-texte-hit]"), rr: bloc.querySelector("[data-texte-rr]") };
    const NOMBRE_RANGS = 10;

    liste.replaceChildren();
    const lignes = Array.from({ length: NOMBRE_RANGS }, (_, i) => {
      const li = document.createElement("li");
      li.className = "rang-ligne" + (i < 3 ? " rang-ligne--top" : "");
      li.innerHTML = '<span class="rang-ligne__num"></span><span class="rang-ligne__texte"></span><svg class="ico" aria-hidden="true"><use href="#i-valider"/></svg>';
      li.querySelector(".rang-ligne__num").textContent = String(i + 1);
      liste.appendChild(li);
      return li;
    });

    let ancien = {};
    const poser = (cle, valeur) => {
      if (ancien[cle] === valeur) return;
      ancien[cle] = valeur;
      champs[cle].textContent = valeur;
      champs[cle].classList.remove("est-change"); void champs[cle].offsetWidth; champs[cle].classList.add("est-change");
    };

    const maj = () => {
      const rang = parseInt(curseur.value, 10);
      const nombre = parseInt(radios.find((r) => r.checked).value, 10);
      const absent = rang > NOMBRE_RANGS;
      sortieRang.textContent = absent ? "absent" : String(rang);
      curseur.setAttribute("aria-valuetext", absent ? "absent des 10 premiers" : `rang ${rang}`);
      const bons = new Set();
      if (!absent) for (let k = 0; k < nombre; k++) if (rang + k <= NOMBRE_RANGS) bons.add(rang + k);
      lignes.forEach((li, i) => {
        const bon = bons.has(i + 1);
        li.classList.toggle("rang-ligne--bon", bon);
        li.querySelector(".rang-ligne__texte").textContent = bon ? "Bon passage" : "Passage sans rapport avec le fait visé";
      });
      const dansTop = [1, 2, 3].filter((r) => bons.has(r)).length;
      poser("p", formatFr(dansTop / 3, 2));
      poser("hit", dansTop > 0 ? "1" : "0");
      poser("rr", absent ? "0" : formatFr(1 / rang, 2));
      textes.p.textContent = nombre === 1
        ? `${dansTop} bon passage sur les 3 affichés. Avec un seul bon passage, Precision@3 ne peut pas dépasser 0,33.`
        : `${dansTop} bon${dansTop > 1 ? "s" : ""} passage${dansTop > 1 ? "s" : ""} sur les 3 affichés : ${dansTop} ÷ 3.`;
      textes.hit.textContent = dansTop > 0 ? "Au moins un bon passage est dans le top 3 : la question est réussie." : "Aucun bon passage dans le top 3 : la question est ratée, même si le bon passage est plus bas.";
      textes.rr.textContent = absent ? "Pas de bon passage dans les 10 premiers : la note est 0." : `1 ÷ ${rang} : plus le bon passage est haut, plus la note est forte. Le MRR est la moyenne de cette note sur toutes les requêtes.`;
    };
    curseur.addEventListener("input", maj);
    radios.forEach((r) => r.addEventListener("change", maj));
    maj();
  });

  /* ---------- comparateur de la page Résultats ---------- */
  document.querySelectorAll("[data-comparateur]").forEach((bloc) => {
    const lignes = [...bloc.querySelectorAll("[data-barres] > li")];
    const nbRequetes = bloc.querySelector("[data-nb-requetes]");
    const NOMBRES = { toutes: 63, meme: 54 };
    const maj = () => {
      const scenario = bloc.querySelector("input[name='scenario']:checked").value;
      const metrique = bloc.querySelector("input[name='metrique']:checked").value;
      bloc.dataset.scenario = scenario; bloc.dataset.metrique = metrique;
      lignes.forEach((ligne) => {
        const valeur = parseFloat(ligne.dataset[`${metrique}${scenario === "toutes" ? "Toutes" : "Meme"}`]);
        ligne.style.setProperty("--v", valeur);
        ligne.querySelector("[data-valeur]").textContent = formatFr(valeur, 2);
      });
      nbRequetes.textContent = String(NOMBRES[scenario]);
    };
    bloc.querySelectorAll("input[type='radio']").forEach((r) => r.addEventListener("change", maj));
  });

  /* ---------- filtre par langue (page Corpus) ---------- */
  document.querySelectorAll("[data-filtre-langue]").forEach((groupe) => {
    const documents = [...document.querySelectorAll("[data-docs] .doc")];
    const vide = document.querySelector("[data-vide]");
    groupe.querySelectorAll("button").forEach((bouton) => {
      bouton.addEventListener("click", () => {
        groupe.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === bouton)));
        const valeur = bouton.dataset.valeur;
        let visibles = 0;
        documents.forEach((doc) => {
          const montrer = valeur === "tous" || doc.dataset.langue === valeur;
          doc.hidden = !montrer;
          if (montrer) { visibles++; doc.style.animation = "none"; void doc.offsetWidth; doc.style.animation = ""; doc.style.setProperty("--i", String(Math.min(visibles, 8))); }
        });
        vide.hidden = visibles > 0;
      });
    });
  });
})();
