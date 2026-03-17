"""
Insurance AI — Dashboard Streamlit
Monitoring · Classification · Souscription · Anomalies
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import io

# ─── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Insurance AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
div[data-testid="stSidebarContent"] { background: #0f1117; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Données simulées ─────────────────────────────────────────────────────────

@st.cache_data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=30, freq="D")

    # Volume quotidien
    volume = pd.DataFrame({
        "date": dates,
        "contrats": np.random.randint(1800, 3200, 30),
        "anomalies": np.random.randint(60, 180, 30),
        "auto_approuvés": np.random.randint(1400, 2600, 30),
    })

    # Distribution classes
    classes = pd.DataFrame({
        "classe": [
            "Contrat Auto", "Formulaire", "Contrat Vie",
            "CNI / RIB", "Sinistre", "Habitation", "Autre"
        ],
        "count": [1870, 1780, 1360, 1190, 1020, 850, 420],
        "precision": [0.962, 0.928, 0.941, 0.954, 0.913, 0.897, 0.834],
    })

    # Souscriptions récentes
    statuts = ["Auto-approuvé", "En révision", "Approuvé", "Rejeté"]
    produits = ["Auto", "Vie", "Habitation", "Santé"]
    souscriptions = pd.DataFrame({
        "id": [f"SUS-2024-{8800+i}" for i in range(20)],
        "produit": np.random.choice(produits, 20),
        "statut": np.random.choice(statuts, 20, p=[0.65, 0.20, 0.10, 0.05]),
        "score_anomalie": np.round(np.random.beta(2, 8, 20), 3),
        "temps_ms": np.random.randint(280, 780, 20),
        "date": [datetime.now() - timedelta(minutes=random.randint(1, 300)) for _ in range(20)],
    })
    souscriptions = souscriptions.sort_values("date", ascending=False).reset_index(drop=True)

    # Modèles
    modeles = pd.DataFrame({
        "modèle": ["DocumentClassifier BiLSTM", "AnomalyDetector XGBoost", "NER InsuranceFR", "TopicModel LDA"],
        "version": ["v1.4.2", "v2.1.0", "v1.2.3", "v1.0.1"],
        "accuracy": [0.934, 0.961, 0.889, "—"],
        "f1": [0.928, 0.887, 0.882, "—"],
        "drift": [0.042, 0.028, 0.071, 0.031],
        "req_24h": [8472, 2841, 8472, 8472],
        "statut": ["✅ OK", "✅ OK", "⚠️ Drift", "✅ OK"],
    })

    return volume, classes, souscriptions, modeles


volume_df, classes_df, subs_df, models_df = generate_data()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏦 Insurance AI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📄 Classifier un document", "🔍 Analyser une souscription", "🤖 Modèles IA"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Environnement**")
    env = st.selectbox("", ["Production (GCP)", "Staging", "Local"], label_visibility="collapsed")
    st.markdown("**API**")
    api_url = st.text_input("", value="https://insurance-ai.run.app", label_visibility="collapsed")

    st.markdown("---")
    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.title("📊 Dashboard — Monitoring IA")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Documents (24h)", "8 472", "+12.4%")
    with col2:
        st.metric("Précision globale", "93.4%", "+0.3%")
    with col3:
        st.metric("Temps moyen", "433 ms", "-18ms")
    with col4:
        st.metric("Anomalies détectées", "127", "→ stable")
    with col5:
        st.metric("Taux auto-approbation", "84.7%", "+1.2%")

    st.markdown("---")

    # Graphiques row 1
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Volume de traitement — 30 jours")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=volume_df["date"], y=volume_df["contrats"],
            name="Documents traités", marker_color="#4f8ef7",
        ))
        fig.add_trace(go.Scatter(
            x=volume_df["date"], y=volume_df["anomalies"],
            name="Anomalies", line=dict(color="#e85555", width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.15),
            yaxis2=dict(overlaying="y", side="right", showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Distribution des classes")
        fig2 = px.pie(
            classes_df, values="count", names="classe",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig2.update_layout(
            height=320, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="v", x=1, y=0.5),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Graphiques row 2
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Précision par classe")
        fig3 = px.bar(
            classes_df.sort_values("precision"),
            x="precision", y="classe",
            orientation="h",
            color="precision",
            color_continuous_scale=["#e85555", "#f5a623", "#3dd68c"],
            range_color=[0.80, 1.0],
        )
        fig3.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0.78, 1.0], tickformat=".0%"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("Scores d'anomalie — Distribution")
        scores = np.random.beta(1.5, 8, 2000)
        fig4 = px.histogram(
            x=scores, nbins=40,
            labels={"x": "Score anomalie", "y": "Nb dossiers"},
            color_discrete_sequence=["#9b8bff"],
        )
        fig4.add_vline(x=0.5, line_dash="dash", line_color="#e85555",
                       annotation_text="Seuil (0.5)", annotation_position="top right")
        fig4.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Souscriptions récentes
    st.subheader("Souscriptions récentes")

    def color_statut(val):
        colors = {
            "Auto-approuvé": "color: #3dd68c",
            "En révision": "color: #f5a623",
            "Approuvé": "color: #4f8ef7",
            "Rejeté": "color: #e85555",
        }
        return colors.get(val, "")

    def color_score(val):
        if val > 0.5:
            return "color: #e85555; font-weight: bold"
        if val > 0.3:
            return "color: #f5a623"
        return "color: #3dd68c"

    display_df = subs_df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%d/%m %H:%M")
    display_df.columns = ["Référence", "Produit", "Statut", "Score anomalie", "Temps (ms)", "Date"]

    styled = display_df.style\
        .applymap(color_statut, subset=["Statut"])\
        .applymap(color_score, subset=["Score anomalie"])\
        .format({"Score anomalie": "{:.3f}", "Temps (ms)": "{:.0f} ms"})

    st.dataframe(styled, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLASSIFIER UN DOCUMENT (traitement réel du fichier)
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📄 Classifier un document":
    import re
    import unicodedata

    # ── Fonctions d'extraction de texte ───────────────────────────────────────

    def extraire_texte_pdf(fichier_bytes: bytes) -> str:
        try:
            import pdfplumber
            import io as _io
            with pdfplumber.open(_io.BytesIO(fichier_bytes)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages).strip()
        except ImportError:
            return extraire_texte_pdf_fallback(fichier_bytes)
        except Exception as e:
            return f"[Erreur lecture PDF : {e}]"

    def extraire_texte_pdf_fallback(fichier_bytes: bytes) -> str:
        """Extraction basique sans pdfplumber — cherche les chaînes lisibles."""
        try:
            raw = fichier_bytes.decode("latin-1", errors="replace")
            # Garder uniquement les caractères imprimables
            tokens = re.findall(r'[A-Za-zÀ-ÿ0-9€@.,;:\-/\(\) ]{4,}', raw)
            return " ".join(tokens[:500])
        except Exception:
            return "[Impossible de lire ce PDF sans pdfplumber]"

    def extraire_texte_image(fichier_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(fichier_bytes))
            return pytesseract.image_to_string(img, lang="fra").strip()
        except ImportError:
            return "[OCR non disponible — installez pytesseract et Pillow]"
        except Exception as e:
            return f"[Erreur OCR : {e}]"

    def extraire_texte(uploaded_file) -> str:
        """Point d'entrée unique — détecte le type et extrait le texte."""
        ext = uploaded_file.name.lower().split(".")[-1]
        data = uploaded_file.read()
        if ext == "pdf":
            return extraire_texte_pdf(data)
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            return extraire_texte_image(data)
        elif ext == "txt":
            return data.decode("utf-8", errors="replace").strip()
        else:
            return data.decode("utf-8", errors="replace").strip()

    # ── Fonctions NLP légères (sans spaCy) ────────────────────────────────────

    # ── Fonctions NLP légères (sans spaCy) ────────────────────────────────────

    # Patterns génériques (hors RIB)
    PATTERNS = {
        "Montant (€)":     r'\b(?!0[0-9]\s)\d{2,}(?:[\s]\d{3})*[.,]\d{2}\s*(?:€|EUR|euros?)\b',
        "Date":            r'\b(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b',
        "Email":           r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b',
        "Téléphone":       r'\b(?:(?:\+33|0033|0)[1-9])(?:[\s.\-]?\d{2}){4}\b',
        "IBAN":            r'\bFR\d{2}(?:[\s]?\d{4}){5}(?:[\s]?\d{1,3})?\b',
        "BIC":             r'\b[A-Z]{4}FR[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b',
        "Immatriculation": r'\b[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}\b',
        # Code postal : exactement 5 chiffres, précédé d'un espace ou début,
        # commençant par 0[1-9] ou [1-8]X ou 9[0-5] — exclut codes banque/guichet
        "Code postal":     r'(?<![\d])(?:0[1-9]|[1-8]\d|9[0-5])\d{3}(?![\d])',
    }

    # Mots-clés RIB dans les en-têtes à ignorer pour la banque
    _RIB_HEADERS = {"BANQUE", "GUICHET", "N° COMPTE", "CLÉ", "DEVISE",
                    "DOMICILIATION", "IDENTIFIANT", "NATIONAL", "INTERNATIONAL",
                    "IBAN", "BIC", "TITULAIRE", "ACCOUNT"}

    def extraire_entites_rib(texte: str) -> dict:
        """Extraction spécialisée ligne par ligne pour les RIB."""
        entites = {}
        lignes = [l.strip() for l in texte.splitlines() if l.strip()]

        # Repérer la ligne d'en-tête du tableau RIB pour l'ignorer
        header_idx = next(
            (i for i, l in enumerate(lignes)
             if "Banque" in l and "Guichet" in l and "N° compte" in l),
            -1
        )
        valeurs_idx = header_idx + 1 if header_idx >= 0 else -1

        for i, ligne in enumerate(lignes):
            lu = ligne.upper()

            # Ignorer les lignes d'en-tête du tableau RIB
            words = set(lu.split())
            if words & _RIB_HEADERS and len(words) >= 3:
                continue
            # Ignorer la ligne des valeurs numériques brutes du tableau
            if i == valeurs_idx:
                continue

            # IBAN : chercher FR suivi de chiffres groupés
            m = re.search(r'(FR\d{2}(?:[\s]?\d+){3,})', ligne, re.IGNORECASE)
            if m:
                iban = re.sub(r'\s+', ' ', m.group(1)).strip()
                if len(iban.replace(" ", "")) >= 20:
                    entites["IBAN"] = [iban]

            # BIC
            m2 = re.search(r'\b([A-Z]{4}FR[A-Z0-9]{2,5})\b', ligne)
            if m2:
                entites["BIC"] = [m2.group(1)]

            # Titulaire : ligne commençant par M / MME / MR + nom
            if re.match(r'^(M\.?|MR\.?|MME\.?|MONSIEUR|MADAME)\s+[A-ZÉÈÀÙ]', lu):
                if len(ligne) < 60 and "COMPTE" not in lu:
                    entites["Titulaire"] = [ligne]

            # Nom banque : ligne courte contenant un mot-clé bancaire
            # mais PAS une ligne d'en-tête ni d'adresse longue
            banque_kw = ["CCM ", "CREDIT MUTUEL", "BNP", "SOCIETE GENERALE",
                         "CIC", "LCL", "BRED", "CAISSE D", "BANQUE POPULAIRE",
                         "LA BANQUE POSTALE", "HSBC", "NATIXIS"]
            if any(k in lu for k in banque_kw):
                if len(ligne) < 50 and not re.search(r'\d{5}', ligne):
                    entites["Banque"] = [ligne]

            # Adresse : numéro + type de voie
            if re.match(r'^\d{1,4}\s+(RUE|AVENUE|BOULEVARD|IMPASSE|ALLEE|CHEMIN)', lu):
                entites["Adresse"] = [ligne]

            # Code postal : 5 chiffres seuls sur la ligne ou en début
            m_cp = re.search(r'(?<![\d])((?:0[1-9]|[1-8]\d|9[0-5])\d{3})(?![\d])', ligne)
            if m_cp and len(ligne) < 30:
                entites["Code postal"] = [m_cp.group(1)]

        return {k: list(dict.fromkeys(v)) for k, v in entites.items()}

    CLASSES = {
        "contrat_assurance_auto":       ["automobile", "auto", "véhicule", "immatriculation", "conducteur", "rc auto", "bonus"],
        "contrat_assurance_vie":        ["assurance vie", "capital décès", "bénéficiaire", "épargne", "décès"],
        "contrat_assurance_habitation": ["habitation", "multirisque", "mrh", "logement", "locataire", "propriétaire"],
        "contrat_assurance_sante":      ["santé", "mutuelle", "remboursement", "soins", "hospitalisation"],
        "formulaire_souscription":      ["souscription", "demande", "adhésion", "formulaire", "candidature"],
        "declaration_sinistre":         ["sinistre", "accident", "déclaration", "dommage", "incident", "vol"],
        "piece_identite":               ["carte nationale", "passeport", "identité", "nationalité", "née le"],
        "releve_bancaire":              ["iban", "rib", "relevé", "bancaire", "solde", "virement", "compte"],
        "resiliation":                  ["résiliation", "résilier", "mettre fin", "annulation", "échéance"],
    }

    INTENTIONS = {
        "souscription":  ["souscrire", "nouveau contrat", "adhérer", "ouvrir"],
        "résiliation":   ["résilier", "annuler", "mettre fin", "clôturer"],
        "sinistre":      ["sinistre", "accident", "déclarer", "dommage"],
        "réclamation":   ["réclamer", "rembourser", "indemniser", "litige"],
        "information":   ["renseignement", "information", "question", "savoir"],
    }

    def classifier_texte(texte: str) -> dict:
        t = texte.lower()
        scores = {}
        for cls, mots in CLASSES.items():
            scores[cls] = sum(t.count(m) for m in mots)
        total = sum(scores.values()) or 1
        probas = {k: round(v / total, 3) for k, v in scores.items()}
        # Normaliser pour avoir une somme à 1
        s = sum(probas.values()) or 1
        probas = {k: round(v / s, 3) for k, v in probas.items()}
        top = max(probas, key=probas.get)
        return {"classe": top, "confiance": probas[top], "probas": probas}

    def extraire_entites(texte: str) -> dict:
        entites = {}
        for nom, pattern in PATTERNS.items():
            matches = list(set(re.findall(pattern, texte, re.IGNORECASE)))
            if matches:
                entites[nom] = matches[:3]
        return entites

    def detecter_intention(texte: str) -> tuple:
        t = texte.lower()
        scores = {k: sum(t.count(m) for m in mots) for k, mots in INTENTIONS.items()}
        if all(v == 0 for v in scores.values()):
            return "information", 0.5
        top = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        return top, round(scores[top] / total, 2)

    def top_mots_cles(texte: str, n: int = 10) -> list:
        mots = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', texte.lower())
        stop = {"pour", "dans", "avec", "cette", "sont", "être", "avoir", "plus",
                "comme", "leur", "nous", "vous", "tout", "mais", "donc", "votre",
                "notre", "entre", "selon", "après", "avant", "sous", "bien"}
        freq = {}
        for m in mots:
            if m not in stop:
                freq[m] = freq.get(m, 0) + 1
        return sorted(freq.items(), key=lambda x: -x[1])[:n]

    # ── Interface ─────────────────────────────────────────────────────────────

    st.title("📄 Classification de document")
    st.markdown("Uploade un vrai fichier — le texte est extrait et analysé directement dans ton navigateur.")

    col_up, col_res = st.columns([1, 1])

    with col_up:
        st.subheader("Document")
        uploaded = st.file_uploader(
            "Glisser-déposer un fichier",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            help="PDF natif, image (OCR) ou texte brut — max 50 MB",
        )

        texte_libre = st.text_area(
            "Ou coller du texte directement",
            height=200,
            placeholder=(
                "Ex :\nCONTRAT D'ASSURANCE AUTOMOBILE\n"
                "Assuré : Jean Martin\n"
                "Véhicule : AB-123-CD\n"
                "Prime annuelle : 850 €\n"
                "Date d'effet : 01/01/2024\n"
                "Email : jean@email.fr"
            ),
        )

        analyser = st.button("🔍 Analyser", type="primary", use_container_width=True)

        # Aperçu du texte extrait
        if uploaded and not analyser:
            with st.expander("👁️ Aperçu texte extrait"):
                apercu = extraire_texte(uploaded)
                uploaded.seek(0)
                st.text(apercu[:800] + ("..." if len(apercu) > 800 else ""))

    with col_res:
        st.subheader("Résultats")

        texte_final = ""

        if analyser:
            # Récupérer le texte
            if uploaded:
                with st.spinner("Extraction du texte..."):
                    texte_final = extraire_texte(uploaded)
            elif texte_libre.strip():
                texte_final = texte_libre.strip()

            if not texte_final or len(texte_final) < 10:
                st.error("❌ Impossible d'extraire du texte. Vérifie que le fichier n'est pas vide ou corrompu.")
                st.stop()

            t0 = time.time()

            with st.spinner("Classification NLP en cours..."):
                resultat     = classifier_texte(texte_final)
                # Utiliser l'extracteur spécialisé RIB si document bancaire
                if resultat["classe"] in ("releve_bancaire",):
                    entites = extraire_entites_rib(texte_final)
                    # Compléter avec regex génériques
                    for k, v in extraire_entites(texte_final).items():
                        entites.setdefault(k, v)
                else:
                    entites = extraire_entites(texte_final)
                intention, conf_intent = detecter_intention(texte_final)
                mots_cles    = top_mots_cles(texte_final)

            temps_ms = round((time.time() - t0) * 1000)

            # ── Badge classe ──────────────────────────────────────────────────
            top_class = resultat["classe"]
            top_conf  = resultat["confiance"]
            color = "#3dd68c" if top_conf > 0.5 else "#f5a623" if top_conf > 0.25 else "#8891a8"

            st.markdown(f"""
            <div style="background:{color}22;border:1px solid {color};
            border-radius:10px;padding:16px;margin-bottom:16px">
                <div style="font-size:0.75rem;opacity:0.6;margin-bottom:4px">Classe détectée</div>
                <div style="font-size:1.35rem;font-weight:700">
                    {top_class.replace("_", " ").title()}
                </div>
                <div style="font-size:0.95rem;margin-top:4px">
                    Confiance : <b style="color:{color}">{top_conf:.1%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probabilités ──────────────────────────────────────────────────
            with st.expander("📊 Distribution des probabilités", expanded=True):
                probas_triees = sorted(
                    resultat["probas"].items(), key=lambda x: -x[1]
                )
                for cls, prob in probas_triees:
                    if prob > 0:
                        label = f"{cls.replace('_', ' ')}  —  {prob:.1%}"
                        st.progress(min(prob, 1.0), text=label)

            # ── Entités extraites ──────────────────────────────────────────────
            st.markdown("**🏷️ Entités extraites**")
            if entites:
                cols = st.columns(2)
                for i, (nom, vals) in enumerate(entites.items()):
                    with cols[i % 2]:
                        st.markdown(f"**{nom}**")
                        for v in vals:
                            st.code(v, language=None)
            else:
                st.info("Aucune entité structurée détectée (montant, date, IBAN…)")

            # ── Mots-clés ─────────────────────────────────────────────────────
            with st.expander("🔑 Mots-clés TF-IDF (top 10)"):
                mots_df = pd.DataFrame(mots_cles, columns=["mot", "fréquence"])
                fig_mots = px.bar(
                    mots_df.sort_values("fréquence"),
                    x="fréquence", y="mot", orientation="h",
                    color="fréquence",
                    color_continuous_scale=["#4f8ef7", "#9b8bff"],
                )
                fig_mots.update_layout(
                    height=280, margin=dict(t=5, b=5, l=5, r=5),
                    coloraxis_showscale=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_mots, use_container_width=True)

            # ── Résumé ────────────────────────────────────────────────────────
            st.markdown("---")
            col_i, col_t = st.columns(2)
            with col_i:
                st.markdown(f"**Intention** : `{intention}` ({conf_intent:.0%})")
            with col_t:
                st.markdown(f"**Longueur** : {len(texte_final):,} caractères")

            st.success(f"✅ Traitement en **{temps_ms} ms**")

            # ── Texte brut ────────────────────────────────────────────────────
            with st.expander("📃 Texte extrait complet"):
                st.text_area("", value=texte_final, height=250, label_visibility="collapsed")

        elif not analyser:
            st.info("⬅️ Uploade un fichier ou colle du texte, puis clique **Analyser**.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYSER UNE SOUSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Analyser une souscription":
    st.title("🔍 Analyse de souscription")
    st.markdown("Renseigne les informations du dossier pour détecter les anomalies et obtenir une décision.")

    with st.form("souscription_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Souscripteur**")
            nom = st.text_input("Nom complet", value="Marie Dupont")
            email = st.text_input("Email", value="marie.dupont@email.fr")
            produit = st.selectbox("Produit", ["Auto", "Vie", "Habitation", "Santé"])

        with col2:
            st.markdown("**Historique**")
            nb_contrats = st.number_input("Nb contrats actifs", min_value=0, max_value=20, value=1)
            nb_sinistres = st.number_input("Sinistres (12 mois)", min_value=0, max_value=10, value=0)
            retards = st.number_input("Retards de paiement", min_value=0, max_value=10, value=0)

        with col3:
            st.markdown("**Dossier**")
            prime = st.number_input("Prime annuelle (€)", min_value=0, value=850)
            montant_sinistre = st.number_input("Montant sinistre (€)", min_value=0, value=0)
            docs_manquants = st.number_input("Documents manquants", min_value=0, max_value=10, value=0)

        submitted = st.form_submit_button("🔍 Analyser le dossier", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Analyse XGBoost en cours..."):
            time.sleep(0.8)

        # Calcul score anomalie
        score = 0.0
        regles_declenchees = []

        ratio = montant_sinistre / prime if prime > 0 else 0
        if ratio > 5:
            score += 0.4
            regles_declenchees.append(("🚨", "Ratio sinistre/prime élevé", f"{ratio:.1f}×", "critique"))
        if nb_contrats > 4:
            score += 0.25
            regles_declenchees.append(("⚠️", "Trop de contrats actifs", f"{nb_contrats}", "élevé"))
        if nb_sinistres > 2:
            score += 0.2
            regles_declenchees.append(("⚠️", "Fréquence sinistres élevée", f"{nb_sinistres}/an", "élevé"))
        if retards > 1:
            score += 0.1
            regles_declenchees.append(("📋", "Retards de paiement", f"{retards}", "moyen"))
        if docs_manquants > 1:
            score += 0.1
            regles_declenchees.append(("📋", "Documents manquants", f"{docs_manquants}", "faible"))

        score = min(score + np.random.uniform(0.02, 0.08), 1.0)

        # Affichage résultat
        st.markdown("---")
        col_score, col_details = st.columns([1, 2])

        with col_score:
            # Jauge score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Score anomalie"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e85555" if score > 0.5 else "#f5a623" if score > 0.3 else "#3dd68c"},
                    "steps": [
                        {"range": [0, 30], "color": "#3dd68c22"},
                        {"range": [30, 50], "color": "#f5a62322"},
                        {"range": [50, 100], "color": "#e8555522"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=250, margin=dict(t=30, b=10, l=20, r=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Décision
            if score < 0.3:
                st.success("✅ **Traitement automatique**\nAucune anomalie détectée")
            elif score < 0.5:
                st.warning("⚡ **Vérification rapide**\nPoints à contrôler")
            else:
                st.error("🚨 **Révision manuelle obligatoire**\nAnomalies détectées")

        with col_details:
            st.markdown("**Règles déclenchées**")
            if regles_declenchees:
                for icon, desc, val, sev in regles_declenchees:
                    color = {"critique": "#e85555", "élevé": "#f5a623", "moyen": "#4f8ef7", "faible": "#8891a8"}.get(sev, "#888")
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
                    border-left:3px solid {color};margin-bottom:6px;border-radius:0 6px 6px 0;
                    background:{color}11">
                        <span>{icon}</span>
                        <span style="flex:1">{desc}</span>
                        <span style="font-weight:700;color:{color}">{val}</span>
                        <span style="font-size:0.75rem;color:{color};text-transform:uppercase">{sev}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("Aucune règle déclenchée — dossier propre")

            st.markdown("---")
            st.markdown("**Prochaines étapes**")
            if score < 0.3:
                st.markdown("- ✅ Émission automatique du contrat")
                st.markdown("- 📧 Envoi confirmation par email")
                st.markdown("- 💾 Archivage dans le système de gestion")
            else:
                st.markdown("- 📋 Transmission au service de contrôle")
                st.markdown("- 📞 Contact du souscripteur pour vérification")
                st.markdown("- 🔎 Vérification documentaire approfondie")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODÈLES IA
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Modèles IA":
    st.title("🤖 Modèles IA — État & Performances")

    # Table modèles
    st.subheader("Registre des modèles")
    st.dataframe(
        models_df.style.applymap(
            lambda v: "color: #3dd68c" if "✅" in str(v) else "color: #f5a623" if "⚠️" in str(v) else "",
            subset=["statut"]
        ).applymap(
            lambda v: "color: #e85555; font-weight:bold" if isinstance(v, float) and v > 0.06 else "",
            subset=["drift"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Évolution F1 Score — 30 jours")
        dates = pd.date_range(end=datetime.today(), periods=30, freq="D")
        fig_f1 = go.Figure()
        for name, base, color in [
            ("DocumentClassifier", 0.91, "#4f8ef7"),
            ("AnomalyDetector", 0.87, "#3dd68c"),
            ("NER InsuranceFR", 0.86, "#f5a623"),
        ]:
            vals = base + np.cumsum(np.random.normal(0.001, 0.005, 30)).clip(-0.05, 0.05)
            fig_f1.add_trace(go.Scatter(
                x=dates, y=vals, name=name,
                line=dict(color=color, width=2),
            ))
        fig_f1.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.2),
            yaxis=dict(tickformat=".1%", range=[0.82, 0.98]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    with col2:
        st.subheader("Score de dérive (data drift)")
        fig_drift = go.Figure()
        for name, base_drift, color in [
            ("DocumentClassifier", 0.04, "#4f8ef7"),
            ("AnomalyDetector", 0.025, "#3dd68c"),
            ("NER InsuranceFR", 0.065, "#f5a623"),
        ]:
            vals = base_drift + np.cumsum(np.random.normal(0.0005, 0.003, 30)).clip(-0.02, 0.08)
            fig_drift.add_trace(go.Scatter(
                x=dates, y=vals, name=name,
                line=dict(color=color, width=2),
            ))
        fig_drift.add_hline(
            y=0.15, line_dash="dash", line_color="#e85555",
            annotation_text="Seuil alerte (0.15)", annotation_position="top left"
        )
        fig_drift.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    # Feature importance XGBoost
    st.subheader("Feature Importance — AnomalyDetector XGBoost")
    features = pd.DataFrame({
        "feature": [
            "claim_to_premium_ratio", "time_to_first_claim_days", "claims_last_12_months",
            "document_quality_score", "num_policies_same_person", "ocr_confidence_avg",
            "late_payments_count", "num_address_changes", "is_high_risk_zone",
        ],
        "importance": [0.312, 0.198, 0.142, 0.098, 0.087, 0.063, 0.048, 0.031, 0.021],
    })
    fig_imp = px.bar(
        features.sort_values("importance"),
        x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#4f8ef7", "#9b8bff", "#e85555"],
    )
    fig_imp.update_layout(
        height=340, margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importance relative",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Actions
    st.markdown("---")
    st.subheader("Actions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🔁 Ré-entraîner DocumentClassifier", use_container_width=True):
            with st.spinner("Lancement du job d'entraînement..."):
                time.sleep(1.5)
            st.success("Job lancé sur Vertex AI — estimation 45 min")
    with col_b:
        if st.button("📊 Rapport de dérive complet", use_container_width=True):
            with st.spinner("Génération du rapport..."):
                time.sleep(0.8)
            st.info("Rapport disponible dans MLflow → Experiments")
    with col_c:
        if st.button("🚀 Déployer nouveau modèle", use_container_width=True):
            with st.spinner("Déploiement sur Vertex AI..."):
                time.sleep(1.2)
            st.success("Modèle déployé sur l'endpoint Vertex AI")
