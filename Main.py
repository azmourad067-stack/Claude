# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from scipy.special import softmax
import math

st.set_page_config(page_title="Turf Quant Pricing Engine", layout="wide", initial_sidebar_state="expanded")

# --- QUANTITATIVE ENGINE FUNCTIONS ---

def parse_musique(musique_str):
    if pd.isna(musique_str) or not isinstance(musique_str, str):
        return 0.0
    musique_str = re.sub(r'\(\d{2}\)', '', musique_str)
    tokens = re.findall(r'(\d+|[DaT])', musique_str)
    score = 0.0
    weight = 1.0
    decay = 0.75 
    for t in tokens:
        if t.isdigit():
            pos = int(t)
            pts = max(0, 12 - pos) if pos > 0 else 0
        else:
            pts = 0
        score += pts * weight
        weight *= decay
    return score

def bayesian_shrinkage(val, prior_mean, prior_weight=5):
    if pd.isna(val):
        return prior_mean
    return (val * 10 + prior_mean * prior_weight) / (10 + prior_weight)

def z_score_normalize(series):
    if series.std() == 0 or pd.isna(series.std()):
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / series.std()

def min_max_normalize(series):
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def compute_implied_probs(odds_series):
    raw_probs = 1.0 / odds_series.replace(0, np.nan)
    overround = raw_probs.sum()
    return raw_probs / overround if overround > 0 else raw_probs

def gumbel_monte_carlo(logits, n_iterations=1000):
    n_horses = len(logits)
    wins = np.zeros(n_horses)
    for _ in range(n_iterations):
        noise = np.random.gumbel(0, 1, size=n_horses)
        winner = np.argmax(logits + noise)
        wins[winner] += 1
    return wins / n_iterations

def generate_combinations(horses, probs, k, num_combos=10):
    combinations = set()
    attempts = 0
    probs_norm = probs / probs.sum()
    while len(combinations) < num_combos and attempts < 2000:
        combo = tuple(np.random.choice(horses, size=k, replace=False, p=probs_norm))
        combinations.add(combo)
        attempts += 1
    return [list(c) for c in combinations]

# --- UI & MAIN APP ---

st.title("🏇 Turf Quant Pricing & Probabilistic Engine")
st.markdown("---")

with st.sidebar:
    st.header("🏁 Paramètres de la Course")
    race_type = st.selectbox("Type de course", ["Plat", "Attelé", "Monté", "Obstacle"])
    distance = st.number_input("Distance (m)", min_value=1000, max_value=6000, value=2100, step=100)
    discipline_level = st.selectbox("Niveau", ["Groupe I", "Groupe II", "Groupe III", "Listed", "Handicap", "Réclamer", "Course à conditions"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Modèle")
    mc_iterations = st.slider("Itérations Monte Carlo", 1000, 10000, 2000, step=1000)

st.header("📥 Saisie des Partants")

default_data = pd.DataFrame({
    "Num": range(1, 11),
    "Sexe": ["M", "F", "H", "M", "M", "F", "H", "H", "M", "F"],
    "Âge": [4, 4, 5, 3, 6, 4, 7, 5, 4, 3],
    "Cote": [2.5, 15.0, 8.5, 4.2, 22.0, 12.0, 35.0, 7.0, 5.5, 18.0],
    "Musique": ["1a 2a 1a", "Da 4a 5a", "2a 1a 3a", "1a 1a", "7a 8a 0a", "3a 4a 2a", "0a Da 9a", "2a 5a 1a", "1a 3a Da", "4a 6a 2a"],
    "Gains": [150000, 45000, 120000, 85000, 25000, 60000, 15000, 95000, 110000, 30000],
    "% Victoire Driver": [15.5, 5.2, 12.0, 18.0, 3.5, 8.0, 2.0, 10.5, 14.0, 6.5],
    "% Victoire Entraîneur": [18.0, 6.0, 14.5, 20.0, 4.0, 9.5, 3.0, 11.0, 16.0, 7.0],
    "Corde": [2, 8, 4, 1, 10, 5, 9, 3, 6, 7] if race_type == "Plat" else [0]*10
})

edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

if st.button("🚀 Analyser la course (Quant Engine)", type="primary"):
    if edited_df.empty or len(edited_df) < 3:
        st.error("Veuillez saisir au moins 3 partants.")
        st.stop()

    with st.spinner("Exécution du modèle probabiliste (Monte Carlo, Softmax, Bayesian Adjustments)..."):
        progress_bar = st.progress(0)
        
        df = edited_df.copy()
        df = df.dropna(subset=['Num', 'Cote'])
        df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce')
        
        # 1. Feature Engineering
        progress_bar.progress(20)
        df['Score_Musique_Raw'] = df['Musique'].apply(parse_musique)
        df['Score_Musique_Norm'] = z_score_normalize(df['Score_Musique_Raw'])
        
        df['Gains_Log'] = np.log1p(pd.to_numeric(df['Gains'], errors='coerce').fillna(0))
        df['Gains_Z'] = z_score_normalize(df['Gains_Log'])
        
        prior_driver = df['% Victoire Driver'].mean() if not pd.isna(df['% Victoire Driver'].mean()) else 8.0
        prior_trainer = df['% Victoire Entraîneur'].mean() if not pd.isna(df['% Victoire Entraîneur'].mean()) else 8.0
        
        df['Driver_Bayes'] = df['% Victoire Driver'].apply(lambda x: bayesian_shrinkage(x, prior_driver))
        df['Trainer_Bayes'] = df['% Victoire Entraîneur'].apply(lambda x: bayesian_shrinkage(x, prior_trainer))
        df['Human_Factor_Z'] = z_score_normalize(df['Driver_Bayes'] * 0.6 + df['Trainer_Bayes'] * 0.4)

        if race_type == "Plat":
            df['Corde_Impact'] = min_max_normalize(pd.to_numeric(df['Corde'], errors='coerce').fillna(df['Corde'].median()))
            df['Corde_Impact'] = 1.0 - df['Corde_Impact']
        else:
            df['Corde_Impact'] = 0.0

        # 2. Dynamic Weighting & Logits
        progress_bar.progress(40)
        w_musique = 0.35
        w_gains = 0.20
        w_human = 0.30
        w_corde = 0.15 if race_type == "Plat" else 0.0
        
        if race_type != "Plat":
            w_musique += 0.05
            w_human += 0.10

        df['Logit_Score'] = (
            df['Score_Musique_Norm'] * w_musique +
            df['Gains_Z'] * w_gains +
            df['Human_Factor_Z'] * w_human +
            df['Corde_Impact'] * w_corde
        )
        
        # 3. Probabilities & Monte Carlo
        progress_bar.progress(60)
        df['Market_Prob'] = compute_implied_probs(df['Cote'])
        
        market_logit = np.log(df['Market_Prob'] / (1 - df['Market_Prob']))
        blended_logit = df['Logit_Score'] * 0.7 + market_logit.fillna(0) * 0.3 
        
        df['Model_Prob_Softmax'] = softmax(blended_logit)
        df['MC_Prob'] = gumbel_monte_carlo(blended_logit.values, n_iterations=mc_iterations)
        
        # Final blended prob (Softmax + MC calibration)
        df['Prob_Finale (%)'] = ((df['Model_Prob_Softmax'] + df['MC_Prob']) / 2) * 100
        df['Prob_Marche (%)'] = df['Market_Prob'] * 100
        
        # 4. Market Pricing & Value Detection
        progress_bar.progress(80)
        df['Value_Ratio'] = df['Prob_Finale (%)'] / df['Prob_Marche (%)']
        df['Edge'] = df['Prob_Finale (%)'] - df['Prob_Marche (%)']
        
        # Sorting
        df = df.sort_values(by='Prob_Finale (%)', ascending=False).reset_index(drop=True)
        
        # Indices
        top_prob = df['Prob_Finale (%)'].iloc[0]
        second_prob = df['Prob_Finale (%)'].iloc[1] if len(df) > 1 else 0
        confiance_idx = min(100, (top_prob - second_prob) * 2 + (top_prob))
        volatilite_idx = -np.sum((df['Prob_Finale (%)']/100) * np.log(df['Prob_Finale (%)']/100 + 1e-9)) * 100 / np.log(len(df))
        
        progress_bar.progress(100)

    # --- RESULTS DISPLAY ---
    st.markdown("---")
    st.header("📊 Résultats de la Modélisation Quantitatives")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Indice de Confiance Global", f"{confiance_idx:.1f}/100", help="Basé sur l'écart de probabilité en tête.")
    col2.metric("Volatilité de la Course", f"{volatilite_idx:.1f}/100", help="Entropie de la distribution des probabilités.")
    val_horses = len(df[df['Value_Ratio'] > 1.2])
    col3.metric("Chevaux Sous-Cotés (Value)", f"{val_horses} détectés")

    # Tableau Principal
    st.subheader("🏆 Classement Probabiliste & Pricing")
    display_df = df[['Num', 'Cote', 'Prob_Finale (%)', 'Prob_Marche (%)', 'Edge', 'Value_Ratio']].copy()
    display_df['Num'] = display_df['Num'].astype(str)
    
    def highlight_value(val):
        color = 'lightgreen' if val > 1.2 else ('lightcoral' if val < 0.8 else '')
        return f'background-color: {color}'
        
    st.dataframe(display_df.style.format({
        'Prob_Finale (%)': '{:.2f}%',
        'Prob_Marche (%)': '{:.2f}%',
        'Edge': '{:+.2f}%',
        'Value_Ratio': '{:.2f}x'
    }).map(highlight_value, subset=['Value_Ratio']), use_container_width=True)

    # Graphiques
    st.markdown("---")
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("📈 Probabilités de Victoire (Modèle)")
        fig1 = px.bar(display_df, x='Num', y='Prob_Finale (%)', text_auto='.2f', color='Prob_Finale (%)', color_continuous_scale='Viridis')
        fig1.update_layout(xaxis_type='category')
        st.plotly_chart(fig1, use_container_width=True)
        
    with colB:
        st.subheader("⚖️ Modèle vs Marché")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=display_df['Prob_Marche (%)'], y=display_df['Prob_Finale (%)'], 
                                  mode='markers+text', text=display_df['Num'], textposition="top center",
                                  marker=dict(size=12, color=display_df['Value_Ratio'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Value"))))
        max_val = max(display_df['Prob_Marche (%)'].max(), display_df['Prob_Finale (%)'].max()) + 5
        fig2.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
        fig2.update_layout(xaxis_title="Probabilité Marché (%)", yaxis_title="Probabilité Modèle (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # Synthèse & Combinaisons
    st.markdown("---")
    st.header("🎯 Sélections & Optimisations Combinatoires")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("🔒 Bases Ultra Solides")
        for num in df['Num'].head(2):
            st.markdown(f"- **Cheval N°{num}**")
            
    with c2:
        st.info("💰 Outsiders à Value Potentielle")
        outsiders = df[(df['Cote'] > 8) & (df['Value_Ratio'] > 1.1)].head(3)
        if outsiders.empty:
            st.markdown("- *Aucun outsider clear value détecté*")
        else:
            for _, row in outsiders.iterrows():
                st.markdown(f"- **Cheval N°{row['Num']}** (Cote: {row['Cote']}, Value: {row['Value_Ratio']:.2f}x)")

    with c3:
        st.warning("📉 Chevaux Surévalués (Lays théoriques)")
        lays = df[df['Value_Ratio'] < 0.7].head(3)
        if lays.empty:
            st.markdown("- *Aucune anomalie majeure de surcote*")
        else:
            for _, row in lays.iterrows():
                st.markdown(f"- **Cheval N°{row['Num']}** (Cote: {row['Cote']})")

    st.markdown("### 🎲 Génération Monte Carlo de Combinaisons")
    horses = df['Num'].values
    probs_mc = df['Prob_Finale (%)'].values / 100.0
    
    col_trio, col_quinte = st.columns(2)
    with col_trio:
        st.markdown("**10 Trios Optimisés**")
        trios = generate_combinations(horses, probs_mc, 3, 10)
        for i, t in enumerate(trios, 1):
            st.code(f"Trio {i}: {t[0]} - {t[1]} - {t[2]}")
            
    with col_quinte:
        st.markdown("**10 Quintés Optimisés**")
        quintes = generate_combinations(horses, probs_mc, 5, 10)
        for i, q in enumerate(quintes, 1):
            st.code(f"Quinté {i}: {q[0]} - {q[1]} - {q[2]} - {q[3]} - {q[4]}")

    st.markdown("---")
    st.markdown("### 🧠 Analyse Automatique Argumentée")
    base1 = df.iloc[0]['Num']
    base2 = df.iloc[1]['Num'] if len(df) > 1 else 'N/A'
    val_top = df.sort_values(by='Value_Ratio', ascending=False).iloc[0]
    
    summary = f"""
    *Note d'analyse quantitative :* La course présente une volatilité de **{volatilite_idx:.1f}**, ce qui indique un marché {'très incertain' if volatilite_idx > 80 else 'modérément ouvert' if volatilite_idx > 60 else 'plutôt lisible'}. 
    Le modèle probabiliste (calibré via Monte Carlo sur {mc_iterations} itérations et ajustement bayésien des facteurs humains) isole clairement le **N°{base1}** avec une probabilité de victoire de **{df.iloc[0]['Prob_Finale (%)']:.1f}%**. 
    En couverture, le **N°{base2}** représente l'alternative la plus robuste statistiquement. 
    L'anomalie de marché la plus forte est identifiée sur le **N°{val_top['Num']}**, qui affiche une probabilité réelle estimée à **{val_top['Prob_Finale (%)']:.1f}%** contre seulement **{val_top['Prob_Marche (%)']:.1f}%** implicite chez les bookmakers, offrant un ratio de value de **{val_top['Value_Ratio']:.2f}x**.
    """
    st.info(summary)
