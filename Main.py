import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.special import softmax
import time
import random
import re

# ==============================================================================
# CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="QuantTurf Pro | Engine Prédictif",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1f2937; text-align: center;}
    .metric-card {background-color: #f3f4f6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #2563eb;}
    .stDataFrame {font-size: 0.9rem;}
    .big-font {font-size: 1.2rem; font-weight: bold;}
    .value-positive {color: #16a34a; font-weight: bold;}
    .value-negative {color: #dc2626; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CORE QUANTITATIVE ENGINE (BACKEND)
# ==============================================================================

class QuantEngine:
    """
    Moteur de calcul probabiliste inspiré des modèles de bookmakers et Hedge Funds.
    Intègre : Bayesian Shrinkage, Exponential Decay, Monte Carlo, Softmax.
    """
    
    def __init__(self):
        self.MUSIC_SCORES = {'1': 10, '2': 7, '3': 5, '4': 3, '5': 1, '0': -5, 'D': -5, 'A': 0}
        
    def parse_music(self, music_str):
        """
        Transforme une chaîne de musique (ex: "1a 2a Da 3a") en un score pondéré.
        Utilise une décroissance exponentielle pour favoriser la forme récente.
        """
        if not music_str or music_str.strip() == "":
            return 0.0, 0.0
        
        clean_music = music_str.upper().replace(" ", "")
        matches = re.findall(r'(\d+|[D])', clean_music)
        
        if not matches:
            return 0.0, 0.5

        scores = []
        for m in matches:
            if m == 'D':
                val = -5
            else:
                try:
                    val = int(m)
                    if val > 5: val = 0
                except:
                    val = 0
            scores.append(val)
        
        decay = 0.85
        weighted_sum = 0
        total_weight = 0
        variance_sum = 0
        
        for i, score in enumerate(scores):
            weight = decay ** i
            weighted_sum += score * weight
            total_weight += weight
            variance_sum += ((score - np.mean(scores))**2) * weight
            
        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        volatility = np.sqrt(variance_sum / total_weight) if total_weight > 0 else 10
        regularity = 1 / (1 + volatility) 
        
        return final_score, regularity

    def bayesian_shrinkage(self, win_rate, n_races, global_avg=0.10):
        """
        Applique un lissage bayésien sur les % de réussite (Driver/Entraîneur).
        """
        if n_races < 5:
            weight_data = n_races / 20.0 
            return (win_rate * weight_data) + (global_avg * (1 - weight_data))
        return win_rate

    def calculate_base_score(self, row, race_params):
        """
        Construit le score brut (Raw Score) avant normalisation.
        """
        score = 0
        
        # 1. Performance Cheval (Musique) - Poids 40%
        music_score, regularity = self.parse_music(row['Musique'])
        score += music_score * 4.0
        
        # 2. Facteurs Humains (Driver/Trainer) - Poids 25%
        driver_eff = self.bayesian_shrinkage(row['% Driver'] / 100.0, 50)
        trainer_eff = self.bayesian_shrinkage(row['% Entraîneur'] / 100.0, 50)
        human_factor = (driver_eff * 0.6) + (trainer_eff * 0.4)
        score += human_factor * 25.0
        
        # 3. Expérience / Gains - Poids 15%
        gains_score = np.log1p(row['Gains']) / 10.0
        score += gains_score * 1.5
        
        # 4. Age/Distance Fit - Poids 10%
        age_penalty = 0
        if race_params['discipline'] == 'Trot':
            if row['Age'] < 4: age_penalty = -2
            elif row['Age'] > 9: age_penalty = -3
        else:
            if row['Age'] < 3: age_penalty = -2
            elif row['Age'] > 6: age_penalty = -3
        score += age_penalty
        
        # 5. Corde (Plat uniquement) - Poids 10%
        if race_params['discipline'] == 'Plat' and row.get('Corde') is not None:
            corde = int(row['Corde'])
            if corde <= 4: score += 2.0
            elif corde >= 14: score -= 1.5
            
        return score, regularity

    def normalize_features(self, df_scores):
        """
        Normalisation Min-Max des scores bruts.
        """
        min_s = df_scores['RawScore'].min()
        max_s = df_scores['RawScore'].max()
        range_s = max_s - min_s
        
        if range_s == 0:
            df_scores['NormScore'] = 0.5
        else:
            df_scores['NormScore'] = (df_scores['RawScore'] - min_s) / range_s
            
        return df_scores

    def run_monte_carlo(self, df, n_simulations=2000):
        """
        Simulation de la course N fois en ajoutant du bruit gaussien.
        """
        results = {i: {'win': 0, 'place': 0, 'show': 0, 'top5': 0} for i in range(len(df))}
        
        means = df['NormScore'].values
        stds = (1.0 - df['Regularity'].values) * 0.3 + 0.05 
        
        for _ in range(n_simulations):
            simulated_perfs = np.random.normal(means, stds)
            simulated_perfs += np.random.uniform(-0.01, 0.01, len(simulated_perfs))
            ranking = np.argsort(-simulated_perfs)
            
            results[ranking[0]]['win'] += 1
            results[ranking[1]]['place'] += 1
            results[ranking[2]]['show'] += 1
            
            for k in range(5):
                results[ranking[k]]['top5'] += 1
                
        df['Prob_Victoire'] = [results[i]['win'] / n_simulations for i in range(len(df))]
        df['Prob_Place'] = [results[i]['place'] / n_simulations for i in range(len(df))]
        df['Prob_Top5'] = [results[i]['top5'] / n_simulations for i in range(len(df))]
        
        return df

    def calculate_market_efficiency(self, df):
        """
        Compare les probabilités du modèle avec les cotes du marché.
        """
        df['Implied_Prob_Raw'] = 1.0 / df['Cote']
        total_implied = df['Implied_Prob_Raw'].sum()
        df['Implied_Prob_True'] = df['Implied_Prob_Raw'] / total_implied
        df['Value_Index'] = (df['Prob_Victoire'] * df['Cote']) - 1.0
        df['Edge'] = df['Prob_Victoire'] - df['Implied_Prob_True']
        
        return df

# ==============================================================================
# STREAMLIT UI & LOGIC
# ==============================================================================

def main():
    st.markdown("<h1 class='main-header'>🏇 QuantTurf Pro | Engine Prédictif</h1>", unsafe_allow_html=True)
    st.markdown("### Module d'Analyse Quantitative & Pricing de Marché")
    
    if 'runners' not in st.session_state:
        st.session_state.runners = []
    if 'race_params' not in st.session_state:
        st.session_state.race_params = {'type': 'Attelé', 'distance': 2700, 'discipline': 'Trot'}

    with st.sidebar:
        st.header("⚙️ Paramètres de la Course")
        col1, col2 = st.columns(2)
        with col1:
            discipline = st.selectbox("Discipline", ["Trot", "Plat", "Obstacle"])
        with col2:
            distance = st.number_input("Distance (m)", min_value=800, max_value=4000, value=2700)
        
        st.session_state.race_params['discipline'] = discipline
        st.session_state.race_params['distance'] = distance
        
        st.divider()
        st.info("💡 Ajoutez les partants un par un ci-dessous.")

    st.subheader("📥 Saisie des Partants")
    
    with st.form("runner_form"):
        c1, c2, c3, c4 = st.columns(4)
        num = c1.number_input("Numéro", min_value=1, max_value=20)
        cote = c2.number_input("Cote", min_value=1.1, max_value=100.0, step=0.1)
        age = c3.number_input("Âge", min_value=2, max_value=15)
        sexe = c4.selectbox("Sexe", ["M", "F", "H"])
        
        c5, c6, c7 = st.columns(3)
        gains = c5.number_input("Gains (€)", min_value=0, value=50000)
        music = c6.text_input("Musique (ex: 1a 2a Da)", value="")
        if discipline == "Plat":
            corde = c7.number_input("Corde", min_value=1, max_value=20, value=0)
        else:
            corde = None
            
        c8, c9 = st.columns(2)
        pct_driver = c8.slider("% Victoire Driver", 0, 100, 15)
        pct_trainer = c9.slider("% Victoire Entraîneur", 0, 100, 10)
        
        submitted = st.form_submit_button("Ajouter le Partant")
        
        if submitted:
            new_runner = {
                'Numéro': num,
                'Cote': cote,
                'Age': age,
                'Sexe': sexe,
                'Gains': gains,
                'Musique': music,
                'Corde': corde,
                '% Driver': pct_driver,
                '% Entraîneur': pct_trainer
            }
            existing = [r for r in st.session_state.runners if r['Numéro'] == num]
            if existing:
                st.session_state.runners = [r for r in st.session_state.runners if r['Numéro'] != num]
            st.session_state.runners.append(new_runner)
            st.success(f"Cheval N°{num} ajouté/mis à jour.")

    if st.session_state.runners:
        df_input = pd.DataFrame(st.session_state.runners)
        st.dataframe(df_input, use_container_width=True, hide_index=True)
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🗑️ Tout effacer", type="secondary"):
                st.session_state.runners = []
                st.rerun()
        with col_btn2:
            analyze_btn = st.button("🚀 LANCER L'ANALYSE QUANTITATIVE", type="primary", use_container_width=True)

        if analyze_btn:
            run_analysis(df_input, st.session_state.race_params)

def run_analysis(df, race_params):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🧮 Calcul des scores bruts et régularité...")
    engine = QuantEngine()
    
    scores = []
    regularities = []
    
    for index, row in df.iterrows():
        s, r = engine.calculate_base_score(row, race_params)
        scores.append(s)
        regularities.append(r)
        
    df['RawScore'] = scores
    df['Regularity'] = regularities
    progress_bar.progress(25)
    
    status_text.text("📏 Normalisation Min-Max des features...")
    df = engine.normalize_features(df)
    progress_bar.progress(40)
    
    status_text.text("🎲 Simulation Monte Carlo (2000 itérations)...")
    time.sleep(0.5) 
    df = engine.run_monte_carlo(df, n_simulations=2000)
    progress_bar.progress(70)
    
    status_text.text("📊 Comparaison Modèle vs Marché (Value Detection)...")
    df = engine.calculate_market_efficiency(df)
    progress_bar.progress(90)
    
    df = df.sort_values(by='Prob_Victoire', ascending=False).reset_index(drop=True)
    
    status_text.text("✅ Analyse Terminée.")
    progress_bar.progress(100)
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    display_results(df, race_params)

def display_results(df, race_params):
    st.divider()
    st.subheader("📈 Résultats de la Modélisation")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    fav_prob = df.iloc[0]['Prob_Victoire'] * 100
    volatility = df['Prob_Victoire'].std() * 100
    value_count = len(df[df['Value_Index'] > 0])
    confidence = min(100, (fav_prob * 1.5) + (value_count * 5))
    
    kpi1.metric("Probabilité Favori", f"{fav_prob:.1f}%", delta="Base Solide" if fav_prob > 25 else "Course Ouverte")
    kpi2.metric("Volatilité Course", f"{volatility:.1f}", delta="Faible" if volatility < 10 else "Élevée")
    kpi3.metric("Value Bets Détectés", value_count)
    kpi4.metric("Indice Confiance Global", f"{confidence:.0f}/100")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏁 Classement & Probabilités", "💰 Value Betting", "🎲 Combinaisons Optimisées", "🧠 Analyse Expert"])
    
    with tab1:
        col_chart, col_table = st.columns([2, 1])
        
        with col_chart:
            fig = px.bar(df, x='Numéro', y='Prob_Victoire', 
                         title="Probabilités de Victoire (Modèle Quantitatif)",
                         labels={'Numéro': 'Cheval', 'Prob_Victoire': 'Probabilité (%)'},
                         color='Prob_Victoire', color_continuous_scale='Viridis')
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
            
        with col_table:
            display_df = df[['Numéro', 'Cote', 'Prob_Victoire', 'Prob_Place', 'RawScore']].copy()
            display_df['Prob_Victoire'] = display_df['Prob_Victoire'].apply(lambda x: f"{x:.1%}")
            display_df['Prob_Place'] = display_df['Prob_Place'].apply(lambda x: f"{x:.1%}")
            display_df.columns = ['N°', 'Cote', 'Win %', 'Place %', 'Score']
            st.dataframe(display_df.style.highlight_max(axis=0, subset=['Score'], color='#d1fae5'), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### 🎯 Détection de Value (Edge > 0)")
        st.markdown("Comparaison entre la probabilité réelle estimée par le modèle et la probabilité implicite des cotes.")
        
        val_df = df.sort_values(by='Value_Index', ascending=False)
        
        fig_scatter = px.scatter(val_df, x='Implied_Prob_True', y='Prob_Victoire', 
                                 size='Cote', hover_name='Numéro',
                                 title="Modèle vs Marché (Au-dessus de la ligne = Value)",
                                 labels={'Implied_Prob_True': 'Probabilité Bookmaker', 'Prob_Victoire': 'Probabilité Modèle'})
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        val_display = val_df[['Numéro', 'Cote', 'Prob_Victoire', 'Implied_Prob_True', 'Value_Index', 'Edge']].copy()
        val_display['Prob_Victoire'] = val_display['Prob_Victoire'].apply(lambda x: f"{x:.2%}")
        val_display['Implied_Prob_True'] = val_display['Implied_Prob_True'].apply(lambda x: f"{x:.2%}")
        val_display['Value_Index'] = val_display['Value_Index'].apply(lambda x: f"{x:.2f}")
        val_display['Edge'] = val_display['Edge'].apply(lambda x: f"{x:.2%}")
        
        def color_value(val):
            try:
                color = '#16a34a' if float(val.replace('%','').replace('+','')) > 0 else '#dc2626'
                return f'color: {color}; font-weight: bold'
            except:
                return ''
            
        st.dataframe(val_display.style.applymap(color_value, subset=['Value_Index', 'Edge']), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### 🎫 Stratégies de Jeu Optimisées")
        
        bases = df.head(2)['Numéro'].tolist()
        st.markdown(f"**🔒 Les 2 Bases Solides :** {', '.join(map(str, bases))}")
        
        outsiders = df[(df['Prob_Victoire'] > 0.05) & (df['Cote'] > 8.0)]['Numéro'].tolist()
        if not outsiders:
            outsiders = df.sort_values(by='Value_Index', ascending=False).head(3)['Numéro'].tolist()
        st.markdown(f"**🚀 3 Outsiders à Value :** {', '.join(map(str, outsiders[:3]))}")
        
        st.markdown("#### 10 Combinaisons Trio (Ordre ou Désordre)")
        trio_combos = generate_combos(df, 'Trio', 10)
        for i, combo in enumerate(trio_combos):
            st.text(f"Trio {i+1}: {combo}")
            
        st.markdown("#### 10 Combinaisons Quinté (Base 2 chevaux + 5 associés)")
        top_7 = df.head(7)['Numéro'].tolist()
        base_q = top_7[:2]
        assoc_q = top_7[2:]
        
        st.text(f"Base Quinté : {base_q[0]} - {base_q[1]}")
        st.text(f"Associés : {assoc_q}")
        st.info("Conseil : Jouer les 2 bases avec les 5 associés en champ réduit.")

    with tab4:
        st.markdown("### 🧠 Synthèse de l'Analyste Quantitatif")
        
        favorite = df.iloc[0]
        second = df.iloc[1] if len(df) > 1 else None
        
        analysis_text = f"""
        **Analyse de la course :**
        Le modèle identifie le **N°{favorite['Numéro']}** comme le cheval le plus cohérent statistiquement. 
        Avec un score brut de **{favorite['RawScore']:.2f}** et une régularité de **{favorite['Regularity']:.2f}**, 
        il affiche une probabilité de victoire de **{favorite['Prob_Victoire']:.1%}**.
        
        """
        if favorite['Value_Index'] > 0:
            analysis_text += f"⚠️ **OPPORTUNITÉ :** Le N°{favorite['Numéro']} est sous-estimé par le marché (Value Index: {favorite['Value_Index']:.2f}). La cote de {favorite['Cote']} offre un rendement positif espéré.\n\n"
        else:
            analysis_text += f"📉 **ATTENTION :** Le favori est légèrement surcoté par le marché. La cote de {favorite['Cote']} ne compense pas totalement le risque.\n\n"
            
        if second:
            analysis_text += f"Le **N°{second['Numéro']}** représente l'opposition directe la plus sérieuse. L'écart de probabilité entre les deux premiers est de **{(favorite['Prob_Victoire'] - second['Prob_Victoire']):.1%}**, ce qui indique une course {'très ouverte' if (favorite['Prob_Victoire'] - second['Prob_Victoire']) < 0.10 else 'hiérarchisée'}."
        
        st.markdown(analysis_text)
        
        st.markdown("**Facteurs clés détectés :**")
        cols = st.columns(3)
        cols[0].metric("Impact Musique", "Élevé" if df['RawScore'].std() > 5 else "Moyen")
        cols[1].metric("Facteur Humain", "Décisif" if (df['% Driver'].max() - df['% Driver'].min()) > 20 else "Neutre")
        cols[2].metric("Spécificité Distance", "À vérifier" if race_params['distance'] > 2000 else "Vitesse pure")

def generate_combos(df, type_bet, n_combos):
    import itertools
    combos = []
    top_horses = df['Numéro'].tolist()
    
    if type_bet == 'Trio':
        pool = top_horses[:6]
        all_trios = list(itertools.permutations(pool, 3))
        random.shuffle(all_trios)
        combos = [f"{t[0]} - {t[1]} - {t[2]}" for t in all_trios[:n_combos]]
    return combos

if __name__ == "__main__":
    main()
