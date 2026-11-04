"""Interface de présentation du projet.  Lancement : streamlit run app.py"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROCESSED = Path("data/processed")
PERIOD_LABELS = {"dev_sept": "Septembre (développement)", "holdout_aug": "Août (holdout)"}
COLORS = {"prix_seul": "#8a8f98", "prix_flow": "#2f6fdb"}

st.set_page_config(page_title="QQQ Options Flow", layout="wide")


@st.cache_data
def load_results():
    return pd.read_parquet(PROCESSED / "results.parquet")


@st.cache_data
def load_day_data():
    return pd.concat(pd.read_parquet(PROCESSED / f"{p}.parquet") for p in PERIOD_LABELS)


@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


if not (PROCESSED / "results.parquet").exists():
    st.error("Résultats introuvables : lance d'abord `python train.py`.")
    st.stop()

results = load_results()

st.title("QQQ Options Flow → mouvement à 5 minutes")
st.caption("Le flow des options 0DTE QQQ annonce-t-il la direction du prix dans les 5 prochaines minutes ?")

tab_res, tab_day, tab_method = st.tabs(["Résultats", "Explorer une journée", "Méthode"])

# ------------------------------------------------------------------ résultats
with tab_res:
    variant = st.radio("Jours inclus", ["tous", "hors_opex"], horizontal=True,
                       format_func={"tous": "Tous les jours",
                                    "hors_opex": "Hors expiration mensuelle"}.get)
    res = results[results["variant"] == variant]

    cols = st.columns(2)
    for col, (period, label) in zip(cols, PERIOD_LABELS.items()):
        r = res[res["period"] == period]
        g = r["gain"]
        col.subheader(label)
        a, b, c = col.columns(3)
        a.metric("AUC prix seul", f"{r['prix_seul'].mean():.3f}")
        b.metric("AUC prix + flow", f"{r['prix_flow'].mean():.3f}")
        c.metric("Apport du flow", f"{g.mean():+.3f}", f"± {g.std() / len(g) ** .5:.3f}",
                 delta_color="off")
        col.caption(f"Le flow aide {(g > 0).sum()} jours sur {len(g)}.")

    st.markdown("#### Apport du flow, jour par jour")
    fig = go.Figure()
    for period, label in PERIOD_LABELS.items():
        r = res[res["period"] == period].sort_values("day")
        fig.add_bar(x=r["day"], y=r["gain"], name=label)
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(yaxis_title="AUC (prix + flow) − AUC (prix seul)", height=380,
                      margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    st.info(
        "**Conclusion.** Sur septembre, ajouter le flow semblait améliorer légèrement le modèle. "
        "Sur août, période jamais regardée pendant le développement, l'effet disparaît. "
        "Le flow d'options 0DTE QQQ, tel que mesuré ici, n'apporte pas d'information "
        "prédictive exploitable sur le mouvement à 5 minutes."
    )

# ------------------------------------------------------------- une journée
with tab_day:
    data = load_day_data()
    bundle = load_model()
    day = st.selectbox("Journée", sorted(data["day"].unique()), index=len(data["day"].unique()) - 1)
    d = data[data["day"] == day].copy()
    d["proba_hausse"] = bundle["model"].predict_proba(d[bundle["features"]])[:, 1]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.45, 0.3, 0.25],
                        subplot_titles=["Prix QQQ", "Direction du flow (prime nette, 15 min)",
                                        "Probabilité de hausse donnée par le modèle"])
    fig.add_scatter(x=d.index, y=d["price"], line_color="#222", row=1, col=1)
    fig.add_bar(x=d.index, y=d["net_premium_share_15m"],
                marker_color=["#2f9e5b" if v > 0 else "#d1495b" for v in d["net_premium_share_15m"]],
                row=2, col=1)
    fig.add_scatter(x=d.index, y=d["proba_hausse"], line_color="#2f6fdb", row=3, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=3, col=1)
    fig.update_layout(height=720, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Prime nette : +1 = tout l'argent va aux calls, −1 = tout va aux puts. "
               "Le modèle affiché est le modèle final entraîné sur toutes les données : "
               "cette vue sert à l'illustration, pas à l'évaluation.")

# ---------------------------------------------------------------- méthode
with tab_method:
    st.markdown("""
**Données** — Massive API : barres minute de QQQ et des options 0DTE à ±1 % du spot,
séance 9h30–16h00 (New York).

**Features** (passé uniquement) — part du volume sur les calls, part nette de la prime,
volume total sur 5/15/30 min, volume récent vs moyenne du jour, rendements 1/5/15 min.

**Target** — 1 si le prix est plus haut 5 minutes plus tard, calculée à l'intérieur de chaque journée.

**Évaluation** — validation glissante : chaque jour est prédit par un modèle entraîné sur les
jours précédents uniquement. Étude d'ablation *prix seul* vs *prix + flow*, mesurée par l'AUC.

**Protocole** — septembre sert au développement ; août est un holdout testé une seule fois,
méthode figée. L'exclusion des jours d'expiration mensuelle a été décidée avant le test.

**Limites** — deux mois de données, horizon très court, pas de coûts de transaction,
QQQ utilisé comme proxy du Nasdaq-100 (NQ).
""")
