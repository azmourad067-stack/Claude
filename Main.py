# app.py

import streamlit as st
import pandas as pd
import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quant Turf Engine", layout="wide")

# =========================
# Utility Functions
# =========================

def parse_musique(musique: str):
    if not musique:
        return []
    cleaned = musique.replace("(", "").replace(")", "").replace("-", "").replace(" ", "")
    results = []
    for c in cleaned:
        if c.isdigit():
            results.append(int(c))
        elif c.upper() == "A":
            results.append(10)
        elif c.upper() == "D":
            results.append(9)
        elif c.upper() == "T":
            results.append(8)
    return results

def exponential_form_score(results, alpha=0.6):
    if len(results) == 0:
        return 0.0
    weights = np.array([alpha**i for i in range(len(results))])
    weights = weights / weights.sum()
    transformed = []
    for r in results:
        if r == 1:
            transformed.append(1.0)
        elif r <= 3:
            transformed.append(0.7)
        elif r <= 5:
            transformed.append(0.4)
        elif r <= 8:
            transformed.append(0.2)
        else:
            transformed.append(0.0)
    return float(np.dot(weights, transformed))

def regularity_score(results):
    if len(results) < 2:
        return 0
    arr = np.array(results)
    return 1 / (1 + np.std(arr))

def minmax_scale(series):
    min_v = series.min()
    max_v = series.max()
    if max_v - min_v == 0:
        return pd.Series(np.ones(len(series))*0.5)
    return (series - min_v) / (max_v - min_v)

def zscore(series):
    std = series.std()
    if std == 0:
        return pd.Series(np.zeros(len(series)))
    return (series - series.mean()) / std

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def monte_carlo_simulation(probs, n_sim=5000):
    wins = np.zeros(len(probs))
    for _ in range(n_sim):
        winner = np.random.choice(len(probs), p=probs)
        wins[winner] += 1
    return wins / n_sim

# =========================
# Feature Engineering
# =========================

def build_features(df, race_type, distance):
    df = df.copy()

    df["form_score"] = df["musique_parsed"].apply(exponential_form_score)
    df["regularity"] = df["musique_parsed"].apply(regularity_score)

    df["age_factor"] = 1 - abs(df["age"] - 5) / 10
    df["gains_exp"] = np.log1p(df["gains"]) / (df["experience"] + 1)

    if race_type == "Plat":
        df["corde_factor"] = 1 - (df["corde"] / df["corde"].max())
    else:
        df["corde_factor"] = 0.5

    df["human_factor"] = (
        0.6 * df["driver_pct"] +
        0.4 * df["trainer_pct"]
    ) / 100

    # Market
    df["market_prob"] = 1 / df["cote"]
    df["market_prob"] = df["market_prob"] / df["market_prob"].sum()

    # Scaling
    for col in ["form_score", "regularity", "age_factor",
                "gains_exp", "corde_factor", "human_factor"]:
        df[col] = minmax_scale(df[col])

    return df

def composite_score(df):
    weights = {
        "form_score": 0.25,
        "regularity": 0.10,
        "age_factor": 0.10,
        "gains_exp": 0.15,
        "corde_factor": 0.05,
        "human_factor": 0.20,
        "market_prob": 0.15
    }
    score = np.zeros(len(df))
    for k, w in weights.items():
        score += df[k] * w
    return score

# =========================
# UI
# =========================

st.title("🏇 Quant Turf Predictive Engine")

with st.form("race_form"):
    col1, col2, col3 = st.columns(3)

    race_type = col1.selectbox("Type de course", ["Plat", "Attelé", "Monté", "Obstacle"])
    distance = col2.number_input("Distance (m)", 800, 6000, 2400)
    discipline = col3.text_input("Discipline")

    n = st.number_input("Nombre de partants", 2, 20, 8)

    st.subheader("Saisie des partants")

    horses = []
    for i in range(int(n)):
        st.markdown(f"### Cheval {i+1}")
        c1, c2, c3, c4 = st.columns(4)
        numero = c1.number_input(f"N°", key=f"num{i}")
        age = c2.number_input("Âge", 2, 12, key=f"age{i}")
        sexe = c3.selectbox("Sexe", ["M", "F", "H"], key=f"sex{i}")
        cote = c4.number_input("Cote", 1.01, 200.0, key=f"cote{i}")

        c5, c6, c7 = st.columns(3)
        musique = c5.text_input("Musique", key=f"mus{i}")
        gains = c6.number_input("Gains", 0.0, 1e7, key=f"gains{i}")
        corde = c7.number_input("Corde", 1, 20, key=f"corde{i}")

        c8, c9 = st.columns(2)
        driver_pct = c8.number_input("% victoire driver", 0.0, 100.0, key=f"driver{i}")
        trainer_pct = c9.number_input("% victoire entraîneur", 0.0, 100.0, key=f"trainer{i}")

        horses.append({
            "numero": numero,
            "age": age,
            "sexe": sexe,
            "cote": cote,
            "musique": musique,
            "gains": gains,
            "corde": corde,
            "driver_pct": driver_pct,
            "trainer_pct": trainer_pct,
            "experience": len(parse_musique(musique)),
            "musique_parsed": parse_musique(musique)
        })

    submitted = st.form_submit_button("Analyser la course")

if submitted:

    progress = st.progress(0)

    df = pd.DataFrame(horses)

    progress.progress(20)

    df = build_features(df, race_type, distance)

    progress.progress(40)

    df["raw_score"] = composite_score(df)

    progress.progress(60)

    df["model_prob_softmax"] = softmax(df["raw_score"].values)

    mc_probs = monte_carlo_simulation(df["model_prob_softmax"].values, 5000)

    df["model_prob_final"] = mc_probs

    progress.progress(80)

    df["value_index"] = df["model_prob_final"] - df["market_prob"]

    df = df.sort_values("model_prob_final", ascending=False).reset_index(drop=True)

    progress.progress(100)

    # Outputs
    st.subheader("📊 Probabilités finales")
    st.dataframe(df[["numero", "model_prob_final", "market_prob", "value_index"]])

    # Plot probabilities
    fig1, ax1 = plt.subplots()
    ax1.bar(df["numero"].astype(str), df["model_prob_final"])
    ax1.set_title("Probabilité Modèle")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(df["numero"].astype(str), df["market_prob"])
    ax2.set_title("Probabilité Marché")
    st.pyplot(fig2)

    # Bases & Outsiders
    bases = df.head(2)["numero"].tolist()
    outsiders = df.sort_values("value_index", ascending=False).head(3)["numero"].tolist()

    st.subheader("🎯 Bases ultra solides")
    st.write(bases)

    st.subheader("💎 Outsiders value")
    st.write(outsiders)

    # Combinaisons
    trio_combos = list(combinations(df["numero"].head(6), 3))[:10]
    quinte_combos = list(combinations(df["numero"].head(8), 5))[:10]

    st.subheader("🔢 10 Combinaisons Trio")
    st.write(trio_combos)

    st.subheader("🔢 10 Combinaisons Quinté")
    st.write(quinte_combos)

    # Indices
    confidence_index = df["model_prob_final"].max()
    volatility_index = df["model_prob_final"].std()

    st.subheader("📈 Indice de confiance")
    st.write(round(confidence_index, 3))

    st.subheader("📉 Indice de volatilité")
    st.write(round(volatility_index, 3))

    # Analyst summary
    top = df.iloc[0]
    st.subheader("🧠 Analyse automatique")
    st.write(
        f"Le cheval {int(top['numero'])} présente la meilleure probabilité "
        f"modélisée à {round(top['model_prob_final']*100,2)}%. "
        f"L'écart avec le marché est de {round(top['value_index']*100,2)} points."
    )
