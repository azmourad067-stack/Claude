import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import time
from scipy.stats import zscore
from scipy.special import softmax

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TurfQuant Pro",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0c10;
    --surface: #111318;
    --surface2: #181b22;
    --border: #252933;
    --gold: #c9a84c;
    --gold2: #f0d080;
    --accent: #4c7bc9;
    --green: #3db87a;
    --red: #e05252;
    --text: #e8eaf0;
    --muted: #7a8090;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--gold);
}

.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-header h1 {
    font-size: 3.2rem;
    letter-spacing: -1px;
    margin: 0;
    background: linear-gradient(135deg, var(--gold), var(--gold2), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-header p {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--gold);
    border-left: 3px solid var(--gold);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'DM Mono', monospace;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--gold2);
    font-family: 'DM Serif Display', serif;
}

.badge-green { color: var(--green); font-weight: 600; }
.badge-red { color: var(--red); font-weight: 600; }
.badge-gold { color: var(--gold); font-weight: 600; }

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
}
.result-card.top {
    border-color: var(--gold);
    box-shadow: 0 0 20px rgba(201,168,76,0.12);
}
.result-card.value {
    border-color: var(--green);
    box-shadow: 0 0 20px rgba(61,184,122,0.10);
}

.analysis-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 0.92rem;
    line-height: 1.75;
    color: var(--text);
    white-space: pre-wrap;
}

.stButton > button {
    background: linear-gradient(135deg, #8b6914, var(--gold), #c9a84c);
    color: #0a0c10;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 2.2rem;
    letter-spacing: 0.5px;
    width: 100%;
    transition: opacity 0.2s;
    font-family: 'DM Sans', sans-serif;
}
.stButton > button:hover { opacity: 0.88; }

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

.stProgress > div > div { background: var(--gold) !important; }

div[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
}

.stSelectbox > div, .stTextInput > div, .stNumberInput > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stTextArea textarea {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

.horse-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}

hr { border-color: var(--border); }

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 8px;
    gap: 0;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    background: var(--surface2) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MUSIC PARSER
# ─────────────────────────────────────────────────────────────
def parse_music(music_str: str) -> dict:
    """
    Parse a horse's music string (e.g. "1a2p3a0p1p") and compute
    a weighted performance score with recency decay.
    Returns a dict of features.
    """
    if not music_str or music_str.strip() == "" or music_str.strip().lower() == "0":
        return {
            "score": 0.0,
            "win_rate": 0.0,
            "place_rate": 0.0,
            "regularity": 0.0,
            "recent_form": 0.0,
            "n_races": 0,
        }

    # Extract positions (ignore letters for discipline suffix)
    positions = re.findall(r'(\d+)', music_str)
    positions = [int(p) for p in positions if p != '']

    if not positions:
        return {
            "score": 0.0,
            "win_rate": 0.0,
            "place_rate": 0.0,
            "regularity": 0.0,
            "recent_form": 0.0,
            "n_races": 0,
        }

    n = len(positions)
    # Exponential recency weights (most recent = highest weight)
    decay = 0.75
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()

    # Score mapping: 1st→10, 2nd→7, 3rd→5, 4th→3, 5th→2, 6th-9th→1, 0/D→0
    def pos_score(p):
        if p == 0:
            return 0.0
        scores = {1: 10, 2: 7, 3: 5, 4: 3, 5: 2}
        return scores.get(p, 1.0)

    raw_scores = np.array([pos_score(p) for p in positions])
    weighted_score = float(np.dot(weights, raw_scores))

    wins = sum(1 for p in positions if p == 1)
    places = sum(1 for p in positions if 0 < p <= 3)
    valid = sum(1 for p in positions if p > 0)

    win_rate = wins / max(valid, 1)
    place_rate = places / max(valid, 1)

    # Regularity = inverse of std of valid positions
    valid_pos = [p for p in positions if p > 0]
    if len(valid_pos) > 1:
        regularity = 1 / (1 + np.std(valid_pos))
    else:
        regularity = 0.5

    # Recent form: average of last 3
    recent = positions[-3:]
    recent_scores = [pos_score(p) for p in recent]
    recent_form = np.mean(recent_scores) if recent_scores else 0.0

    return {
        "score": weighted_score,
        "win_rate": win_rate,
        "place_rate": place_rate,
        "regularity": regularity,
        "recent_form": recent_form,
        "n_races": n,
    }


# ─────────────────────────────────────────────────────────────
# PROBABILISTIC ENGINE
# ─────────────────────────────────────────────────────────────
def compute_implicit_prob(cote: float) -> float:
    """Convert bookmaker odds to implicit probability."""
    if cote <= 1.0:
        return 0.99
    return 1.0 / cote


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.ones_like(arr) / len(arr)
    return (arr - arr.min()) / rng


def zscore_clip(arr: np.ndarray) -> np.ndarray:
    if arr.std() == 0:
        return np.zeros_like(arr)
    z = (arr - arr.mean()) / arr.std()
    return np.clip(z, -3, 3)


def bayesian_update(prior: np.ndarray, evidence: np.ndarray, weight: float = 0.3) -> np.ndarray:
    """Simple Bayesian update: blend prior with likelihood evidence."""
    posterior = (1 - weight) * prior + weight * evidence
    return posterior / posterior.sum()


def monte_carlo_simulation(scores: np.ndarray, n_iter: int = 2000) -> np.ndarray:
    """
    Monte Carlo race simulation.
    Each iteration adds Gaussian noise proportional to variance,
    then softmax → argmin to pick winner.
    Returns win frequency per horse.
    """
    n = len(scores)
    wins = np.zeros(n)
    noise_std = np.std(scores) * 0.35 + 1e-6

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        noisy = scores + rng.normal(0, noise_std, n)
        winner = np.argmax(noisy)
        wins[winner] += 1

    return wins / n_iter


def compute_composite_score(horses: list, race_type: str, distance: int) -> dict:
    """
    Core quant engine. Returns per-horse feature matrix and probabilities.
    """
    n = len(horses)
    if n == 0:
        return {}

    # ── Feature extraction ──────────────────────────────────
    music_scores    = np.array([h["music"]["score"] for h in horses])
    recent_form     = np.array([h["music"]["recent_form"] for h in horses])
    win_rate_music  = np.array([h["music"]["win_rate"] for h in horses])
    place_rate      = np.array([h["music"]["place_rate"] for h in horses])
    regularity      = np.array([h["music"]["regularity"] for h in horses])
    n_races         = np.array([h["music"]["n_races"] for h in horses])

    driver_pct  = np.array([h["driver_pct"] / 100.0 for h in horses])
    trainer_pct = np.array([h["trainer_pct"] / 100.0 for h in horses])
    gains       = np.array([h["gains"] for h in horses])
    ages        = np.array([h["age"] for h in horses])
    cotes       = np.array([h["cote"] if h["cote"] > 1.0 else 99.0 for h in horses])
    corde       = np.array([h.get("corde", 0) for h in horses])

    # ── Market probabilities ────────────────────────────────
    raw_market = np.array([compute_implicit_prob(c) for c in cotes])
    # Remove bookmaker margin → normalize
    market_prob = raw_market / raw_market.sum()

    # ── Age / distance optimal mapping ─────────────────────
    # Shorter distances: younger is better; longer: experienced
    # Attelé/monté → age matters less
    if race_type.lower() == "plat":
        if distance <= 1400:
            age_score = np.array([1.0 if 3 <= a <= 5 else 0.6 for a in ages])
        elif distance <= 2000:
            age_score = np.array([1.0 if 4 <= a <= 6 else 0.7 for a in ages])
        else:
            age_score = np.array([1.0 if 5 <= a <= 8 else 0.65 for a in ages])
    else:
        age_score = np.array([1.0 if 4 <= a <= 9 else 0.7 for a in ages])

    # ── Rope (corde) effect in flat racing ─────────────────
    if race_type.lower() == "plat" and corde.max() > 0:
        # Optimal 3-7; extremes penalized
        rope_score = np.array([
            1.0 if 3 <= c <= 7 else
            (0.85 if c in [1, 2, 8, 9] else 0.70)
            for c in corde
        ])
    else:
        rope_score = np.ones(n)

    # ── Gains → experience ratio ────────────────────────────
    exp_ratio = gains / (np.maximum(n_races, 1))  # gains per race
    exp_norm  = minmax_norm(exp_ratio)

    # ── Jockey / trainer combo ─────────────────────────────
    human_factor = (0.55 * driver_pct + 0.45 * trainer_pct)
    # Interaction: horse form × human factor
    interaction  = (music_scores / (music_scores.max() + 1e-6)) * human_factor

    # ── Z-score standardization ─────────────────────────────
    z_music    = zscore_clip(music_scores)
    z_recent   = zscore_clip(recent_form)
    z_human    = zscore_clip(human_factor)
    z_exp      = zscore_clip(exp_norm)
    z_reg      = zscore_clip(regularity)
    z_winrate  = zscore_clip(win_rate_music)

    # ── Dynamic weighting by race type ─────────────────────
    weights = {
        "plat":     {"music": 0.30, "recent": 0.18, "human": 0.18, "exp": 0.10,
                     "age": 0.08, "reg": 0.07, "winrate": 0.07, "rope": 0.02},
        "attelé":   {"music": 0.28, "recent": 0.20, "human": 0.22, "exp": 0.10,
                     "age": 0.06, "reg": 0.08, "winrate": 0.06, "rope": 0.00},
        "monté":    {"music": 0.28, "recent": 0.18, "human": 0.20, "exp": 0.10,
                     "age": 0.08, "reg": 0.08, "winrate": 0.06, "rope": 0.02},
        "obstacle": {"music": 0.25, "recent": 0.20, "human": 0.18, "exp": 0.12,
                     "age": 0.10, "reg": 0.08, "winrate": 0.05, "rope": 0.02},
        "default":  {"music": 0.28, "recent": 0.18, "human": 0.20, "exp": 0.10,
                     "age": 0.08, "reg": 0.08, "winrate": 0.06, "rope": 0.02},
    }
    w = weights.get(race_type.lower(), weights["default"])

    composite = (
        w["music"]   * minmax_norm(z_music + 3)  +
        w["recent"]  * minmax_norm(z_recent + 3) +
        w["human"]   * minmax_norm(z_human + 3)  +
        w["exp"]     * exp_norm                   +
        w["age"]     * age_score                  +
        w["reg"]     * minmax_norm(z_reg + 3)     +
        w["winrate"] * minmax_norm(z_winrate + 3) +
        w["rope"]    * rope_score
    )

    # ── Bayesian update: blend model priors with market ─────
    model_prior  = softmax(composite * 5)
    market_prior = market_prob
    bayesian_prob = bayesian_update(model_prior, market_prior, weight=0.20)

    # ── Monte Carlo simulation ──────────────────────────────
    mc_prob = monte_carlo_simulation(composite * 10)

    # ── Final probability fusion ────────────────────────────
    final_prob = 0.55 * bayesian_prob + 0.45 * mc_prob
    final_prob = final_prob / final_prob.sum()

    # ── Calibration (Platt scaling approximation) ───────────
    # Logit transform then re-softmax for calibration
    eps = 1e-6
    logit_p = np.log(final_prob + eps) - np.log(1 - final_prob + eps)
    calibrated = softmax(logit_p)
    calibrated = calibrated / calibrated.sum()

    # ── Value detection ─────────────────────────────────────
    value_index = calibrated - market_prob  # positive = undervalued
    expected_value = np.array([
        (calibrated[i] * (cotes[i] - 1) - (1 - calibrated[i]))
        for i in range(n)
    ])

    # ── Confidence & volatility ─────────────────────────────
    entropy = -np.sum(calibrated * np.log(calibrated + eps))
    max_entropy = np.log(n)
    confidence_idx = float(1 - entropy / max_entropy) * 100

    # Volatility = spread of model vs market
    volatility_idx = float(np.mean(np.abs(calibrated - market_prob)) * 100)

    return {
        "composite":     composite,
        "model_prob":    calibrated,
        "market_prob":   market_prob,
        "value_index":   value_index,
        "expected_value": expected_value,
        "confidence":    confidence_idx,
        "volatility":    volatility_idx,
        "mc_prob":       mc_prob,
        "human_factor":  human_factor,
        "age_score":     age_score,
        "music_score":   music_scores,
    }


def generate_combinations(ranking: list, model_prob: np.ndarray) -> dict:
    """Generate Trio and Quinté combinations."""
    from itertools import combinations, permutations

    nums = [h["numero"] for h in ranking]
    probs = model_prob

    # Trio: top 3 from top 6
    top6 = nums[:6]
    trio_combos = []
    for combo in combinations(top6, 3):
        idx = [nums.index(c) for c in combo]
        p = np.prod([probs[i] for i in idx])
        trio_combos.append((combo, p))
    trio_combos.sort(key=lambda x: -x[1])

    # Quinté: top 5 from top 8
    top8 = nums[:8]
    quinte_combos = []
    for combo in combinations(top8, 5):
        idx = [nums.index(c) for c in combo]
        p = np.prod([probs[i] for i in idx])
        quinte_combos.append((combo, p))
    quinte_combos.sort(key=lambda x: -x[1])

    return {
        "trio":   trio_combos[:10],
        "quinte": quinte_combos[:10],
    }


def generate_analysis(horses: list, ranking: list, results: dict, race_info: dict) -> str:
    """Generate professional analyst-style race commentary."""
    n = len(horses)
    model_prob = results["model_prob"]
    market_prob = results["market_prob"]
    value_index = results["value_index"]
    conf = results["confidence"]
    vol  = results["volatility"]

    top1 = ranking[0]
    top2 = ranking[1] if n > 1 else None
    base1 = ranking[0]
    base2 = ranking[1] if n > 1 else None

    # Value bets: positive value_index AND cote > 3
    value_horses = [
        h for h in ranking
        if results["value_index"][ranking.index(h)] > 0.025
        and h["cote"] > 3.0
    ][:3]

    lines = []
    lines.append(f"═══ ANALYSE QUANTITATIVE — {race_info['type'].upper()} · {race_info['distance']}m · {n} PARTANTS ═══\n")
    lines.append(f"📊 INDICE DE CONFIANCE : {conf:.1f}/100   |   📈 VOLATILITÉ : {vol:.1f}%\n")

    lines.append(f"\n🏆 FAVORI MODÈLE : N°{top1['numero']} — {top1['nom'].upper()}")
    lines.append(f"   Probabilité modèle : {model_prob[0]*100:.1f}%  |  Probabilité marché : {market_prob[0]*100:.1f}%")
    p_diff = (model_prob[0] - market_prob[0]) * 100
    if p_diff > 2:
        lines.append(f"   ✅ SOUS-COTÉ de {p_diff:.1f}pts — signal favorable confirmé par le modèle")
    elif p_diff < -2:
        lines.append(f"   ⚠️  SURÉVALUÉ de {abs(p_diff):.1f}pts — marché plus optimiste que le modèle")
    else:
        lines.append(f"   ↔️  Alignement modèle/marché ({p_diff:+.1f}pts) — cote juste")

    music1 = top1["music"]
    lines.append(f"   Musique : {top1['musique']} → Score pondéré {music1['score']:.2f}, taux victoire {music1['win_rate']*100:.0f}%")
    lines.append(f"   Driver : {top1['driver_pct']}% | Entraîneur : {top1['trainer_pct']}% | Âge : {top1['age']} ans | Gains : {top1['gains']:,}€")

    if top2:
        lines.append(f"\n🥈 2ème PROBABILISTE : N°{top2['numero']} — {top2['nom'].upper()}")
        lines.append(f"   Probabilité modèle : {model_prob[1]*100:.1f}%  |  Cote : {top2['cote']}")

    lines.append(f"\n🎯 BASES SOLIDES :")
    lines.append(f"   BASE 1 → N°{base1['numero']} {base1['nom']} ({model_prob[0]*100:.1f}%)")
    if base2:
        lines.append(f"   BASE 2 → N°{base2['numero']} {base2['nom']} ({model_prob[1]*100:.1f}%)")

    if value_horses:
        lines.append(f"\n💎 OUTSIDERS VALUE (sous-côtés) :")
        for vh in value_horses:
            idx = ranking.index(vh)
            vi  = value_index[idx]
            lines.append(f"   N°{vh['numero']} {vh['nom']} — Cote {vh['cote']} | Value Index +{vi*100:.1f}pts | EV {results['expected_value'][idx]:.3f}")

    lines.append(f"\n🧠 LECTURE DU MARCHÉ :")
    over  = [h for h in ranking if results["value_index"][ranking.index(h)] < -0.03]
    under = [h for h in ranking if results["value_index"][ranking.index(h)] > 0.03]
    if over:
        over_str = ', '.join([f"N°{h['numero']}" for h in over[:3]])
        lines.append(f"   Surcôtés (marché surévalue) : {over_str}")
    if under:
        under_str = ', '.join([f"N°{h['numero']}" for h in under[:3]])
        lines.append(f"   Sous-côtés (value potentielle) : {under_str}")

    lines.append(f"\n📉 VOLATILITÉ COURSE :")
    if vol < 8:
        lines.append("   Course lisible — faible dispersion des probabilités. Favori nettement dominant.")
    elif vol < 15:
        lines.append("   Course ouverte — plusieurs chevaux dans un mouchoir. Outsiders à surveiller.")
    else:
        lines.append("   Course très ouverte / incertaine — forte volatilité. Stratégie de couverture recommandée.")

    lines.append(f"\n⚡ NOTE ANALYSTE :")
    if conf > 65:
        lines.append(f"   Haute confiance dans les prédictions (score {conf:.1f}). Le modèle détecte un signal clair.")
    elif conf > 45:
        lines.append(f"   Confiance modérée (score {conf:.1f}). Jouer les bases avec un outsider value.")
    else:
        lines.append(f"   Confiance faible (score {conf:.1f}). Course très ouverte — privilégier les combinaisons larges.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "horses" not in st.session_state:
    st.session_state.horses = []
if "results" not in st.session_state:
    st.session_state.results = None


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🏇 TurfQuant Pro</h1>
    <p>Moteur Prédictif Quantitatif · Modélisation Probabiliste Avancée · Grade Bookmaker</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    # ── Race Info ───────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Informations Course</div>', unsafe_allow_html=True)

    with st.container():
        race_type = st.selectbox(
            "Type de course",
            ["Plat", "Attelé", "Monté", "Obstacle", "Cross", "Steeple"],
            key="race_type"
        )
        col_d, col_dis = st.columns(2)
        with col_d:
            distance = st.number_input("Distance (m)", min_value=800, max_value=6000, value=1600, step=100)
        with col_dis:
            discipline = st.selectbox("Discipline", ["Trot", "Galop", "Obstacle"])

        race_level = st.selectbox(
            "Niveau / Catégorie",
            ["Inconnu", "Groupe 1", "Groupe 2", "Groupe 3", "Listed", "Classique",
             "National", "Régional", "Apprentis", "Amateurs", "Réclamer"]
        )

    # ── Add Horse ───────────────────────────────────────────
    st.markdown('<div class="section-title">🐎 Ajouter un Partant</div>', unsafe_allow_html=True)

    with st.form("add_horse_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            h_num    = st.number_input("N° Cheval", min_value=1, max_value=30, value=len(st.session_state.horses)+1)
            h_nom    = st.text_input("Nom du cheval", placeholder="MIDNIGHT STAR")
            h_sexe   = st.selectbox("Sexe", ["H", "F", "G", "E", "M"])
            h_age    = st.number_input("Âge", min_value=2, max_value=20, value=5)
            h_corde  = st.number_input("N° Corde", min_value=0, max_value=30, value=0,
                                        help="0 = non applicable")
        with c2:
            h_cote    = st.number_input("Cote (0=NR)", min_value=0.0, max_value=999.0, value=5.0, step=0.1)
            h_musique = st.text_area("Musique", placeholder="1a2p3a0p", height=70,
                                     help="Ex: 1a2p3p (positions récentes, plus récent à droite)")
            h_gains   = st.number_input("Gains (€)", min_value=0, value=15000, step=1000)
            h_driver  = st.number_input("% Victoires Driver/Jockey", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
            h_trainer = st.number_input("% Victoires Entraîneur", min_value=0.0, max_value=100.0, value=10.0, step=0.5)

        submitted = st.form_submit_button("➕ Ajouter le partant")
        if submitted:
            horse = {
                "numero":     h_num,
                "nom":        h_nom if h_nom.strip() else f"CHEVAL {h_num}",
                "sexe":       h_sexe,
                "age":        h_age,
                "cote":       h_cote if h_cote > 0 else 99.0,
                "musique":    h_musique,
                "gains":      h_gains,
                "driver_pct": h_driver,
                "trainer_pct": h_trainer,
                "corde":      h_corde,
                "music":      parse_music(h_musique),
            }
            # Check for duplicate numero
            existing = [i for i, h in enumerate(st.session_state.horses) if h["numero"] == h_num]
            if existing:
                st.session_state.horses[existing[0]] = horse
                st.success(f"✅ N°{h_num} mis à jour")
            else:
                st.session_state.horses.append(horse)
                st.success(f"✅ N°{h_num} ajouté — Total : {len(st.session_state.horses)} partants")

    # ── Clear & Count ───────────────────────────────────────
    if st.session_state.horses:
        c_count, c_clear = st.columns(2)
        with c_count:
            st.markdown(f"""
            <div class="metric-card" style="padding:0.7rem">
                <div class="label">Partants enregistrés</div>
                <div class="value">{len(st.session_state.horses)}</div>
            </div>""", unsafe_allow_html=True)
        with c_clear:
            if st.button("🗑️ Vider la liste"):
                st.session_state.horses = []
                st.session_state.results = None
                st.rerun()

        # ── Horse table ────────────────────────────────────
        st.markdown('<div class="section-title">📋 Partants enregistrés</div>', unsafe_allow_html=True)
        df_horses = pd.DataFrame([{
            "N°":       h["numero"],
            "Nom":      h["nom"],
            "Sexe":     h["sexe"],
            "Âge":      h["age"],
            "Cote":     h["cote"],
            "Musique":  h["musique"],
            "Gains €":  h["gains"],
            "Driver%":  h["driver_pct"],
            "Trainer%": h["trainer_pct"],
            "Corde":    h["corde"],
        } for h in st.session_state.horses]).sort_values("N°")
        st.dataframe(df_horses, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# RIGHT PANEL — RESULTS
# ─────────────────────────────────────────────────────────────
with col_right:
    # ── Analyze Button ──────────────────────────────────────
    if len(st.session_state.horses) >= 2:
        if st.button("🔬 ANALYSER LA COURSE", use_container_width=True):
            horses = st.session_state.horses
            race_info = {
                "type":       race_type,
                "distance":   distance,
                "discipline": discipline,
                "level":      race_level,
                "n":          len(horses),
            }

            progress = st.progress(0)
            status   = st.empty()

            status.text("🔄 Extraction des features musicales...")
            time.sleep(0.3)
            progress.progress(15)

            status.text("📊 Normalisation Z-score & Min-Max...")
            time.sleep(0.2)
            progress.progress(30)

            results = compute_composite_score(horses, race_type, distance)
            progress.progress(55)
            status.text("🎲 Simulation Monte Carlo (2000 itérations)...")
            time.sleep(0.4)
            progress.progress(75)

            status.text("🧠 Calibration probabiliste & ajustement bayésien...")
            time.sleep(0.3)
            progress.progress(88)

            # Build ranked list
            order = np.argsort(results["model_prob"])[::-1]
            ranking = [horses[i] for i in order]
            model_prob_ranked   = results["model_prob"][order]
            market_prob_ranked  = results["market_prob"][order]
            value_index_ranked  = results["value_index"][order]
            ev_ranked           = results["expected_value"][order]

            combos = generate_combinations(ranking, model_prob_ranked)
            analysis_text = generate_analysis(horses, ranking, {
                "model_prob":    model_prob_ranked,
                "market_prob":   market_prob_ranked,
                "value_index":   value_index_ranked,
                "expected_value": ev_ranked,
                "confidence":    results["confidence"],
                "volatility":    results["volatility"],
            }, race_info)

            progress.progress(100)
            status.empty()
            progress.empty()

            st.session_state.results = {
                "ranking":      ranking,
                "results":      results,
                "order":        order,
                "combos":       combos,
                "analysis":     analysis_text,
                "race_info":    race_info,
                "model_prob_ranked":  model_prob_ranked,
                "market_prob_ranked": market_prob_ranked,
                "value_index_ranked": value_index_ranked,
                "ev_ranked":          ev_ranked,
            }
            st.rerun()
    else:
        st.info("➕ Ajoutez au moins 2 partants pour lancer l'analyse.")

    # ─────────────────────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────────────────────
    if st.session_state.results:
        R         = st.session_state.results
        ranking   = R["ranking"]
        results   = R["results"]
        model_p   = R["model_prob_ranked"]
        market_p  = R["market_prob_ranked"]
        value_idx = R["value_index_ranked"]
        ev        = R["ev_ranked"]
        combos    = R["combos"]
        n         = len(ranking)

        # ── KPI Cards ────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Partants</div>
                <div class="value">{n}</div></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Confiance</div>
                <div class="value">{results['confidence']:.0f}<span style="font-size:1rem">/100</span></div></div>""",
                unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Volatilité</div>
                <div class="value">{results['volatility']:.1f}<span style="font-size:1rem">%</span></div></div>""",
                unsafe_allow_html=True)
        with k4:
            top_p = model_p[0] * 100
            st.markdown(f"""<div class="metric-card">
                <div class="label">Top favori</div>
                <div class="value">{top_p:.1f}<span style="font-size:1rem">%</span></div></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["📊 Classement", "📈 Graphiques", "🎯 Combinaisons", "🧠 Analyse"])

        # ── TAB 1: Ranking ───────────────────────────────────
        with tabs[0]:
            st.markdown('<div class="section-title">Classement Probabiliste</div>', unsafe_allow_html=True)

            df_rank = pd.DataFrame([{
                "Rang":        i + 1,
                "N°":         h["numero"],
                "Cheval":     h["nom"],
                "Cote":       h["cote"] if h["cote"] < 99 else "NR",
                "Prob Modèle": f"{model_p[i]*100:.1f}%",
                "Prob Marché": f"{market_p[i]*100:.1f}%",
                "Value Index": f"{value_idx[i]*100:+.1f}pts",
                "EV":          f"{ev[i]:.3f}",
                "Score Music": f"{h['music']['score']:.2f}",
                "W% Music":    f"{h['music']['win_rate']*100:.0f}%",
                "Driver%":     h["driver_pct"],
                "Trainer%":    h["trainer_pct"],
                "Âge":        h["age"],
                "Musique":    h["musique"],
            } for i, h in enumerate(ranking)])

            # Color code value index
            def style_value(val):
                try:
                    v = float(val.replace("pts", "").replace("+", ""))
                    if v > 3:
                        return "color: #3db87a; font-weight: 600"
                    elif v < -3:
                        return "color: #e05252"
                    return ""
                except:
                    return ""

            st.dataframe(
                df_rank,
                use_container_width=True,
                hide_index=True,
                height=min(40 + n * 36, 500)
            )

            # Bases & outsiders
            st.markdown('<div class="section-title">🏆 Sélection Stratégique</div>', unsafe_allow_html=True)
            b1c, b2c = st.columns(2)
            with b1c:
                h0 = ranking[0]
                vi_str = f"{value_idx[0]*100:+.1f}pts"
                st.markdown(f"""<div class="result-card top">
                    <div style="font-size:.72rem;color:#7a8090;letter-spacing:1.5px;font-family:'DM Mono',monospace">BASE SOLIDE #1</div>
                    <div style="font-size:1.4rem;font-family:'DM Serif Display',serif;color:#f0d080">N°{h0['numero']} {h0['nom']}</div>
                    <div style="margin-top:.4rem">Prob: <strong>{model_p[0]*100:.1f}%</strong> &nbsp;|&nbsp; Cote: <strong>{h0['cote']}</strong> &nbsp;|&nbsp; Value: <span class="{'badge-green' if value_idx[0]>0 else 'badge-red'}">{vi_str}</span></div>
                </div>""", unsafe_allow_html=True)
            with b2c:
                if n > 1:
                    h1 = ranking[1]
                    vi_str = f"{value_idx[1]*100:+.1f}pts"
                    st.markdown(f"""<div class="result-card top">
                        <div style="font-size:.72rem;color:#7a8090;letter-spacing:1.5px;font-family:'DM Mono',monospace">BASE SOLIDE #2</div>
                        <div style="font-size:1.4rem;font-family:'DM Serif Display',serif;color:#f0d080">N°{h1['numero']} {h1['nom']}</div>
                        <div style="margin-top:.4rem">Prob: <strong>{model_p[1]*100:.1f}%</strong> &nbsp;|&nbsp; Cote: <strong>{h1['cote']}</strong> &nbsp;|&nbsp; Value: <span class="{'badge-green' if value_idx[1]>0 else 'badge-red'}">{vi_str}</span></div>
                    </div>""", unsafe_allow_html=True)

            # Value outsiders
            st.markdown("**💎 Outsiders à Value Potentielle**")
            value_horses = [
                (i, h) for i, h in enumerate(ranking)
                if value_idx[i] > 0.02 and h["cote"] > 3.0 and i > 1
            ][:3]
            if value_horses:
                for vi_rank, (vi_i, vh) in enumerate(value_horses):
                    st.markdown(f"""<div class="result-card value">
                        <span style="font-size:.7rem;color:#7a8090;font-family:'DM Mono',monospace">OUTSIDER VALUE #{vi_rank+1}</span>
                        &nbsp;<strong>N°{vh['numero']} {vh['nom']}</strong>
                        &nbsp;|&nbsp; Cote: {vh['cote']}
                        &nbsp;|&nbsp; Prob Modèle: {model_p[vi_i]*100:.1f}%
                        &nbsp;|&nbsp; <span class="badge-green">Value +{value_idx[vi_i]*100:.1f}pts</span>
                        &nbsp;|&nbsp; EV: {ev[vi_i]:.3f}
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("*Aucun outsider value significatif détecté dans cette course.*")

        # ── TAB 2: Charts ────────────────────────────────────
        with tabs[1]:
            fig_prob = go.Figure()

            names = [f"N°{h['numero']} {h['nom'][:12]}" for h in ranking]
            colors_bar = ["#c9a84c" if i < 2 else "#4c7bc9" if value_idx[i] > 0.02 else "#3a4055"
                          for i in range(n)]

            fig_prob.add_trace(go.Bar(
                x=names,
                y=model_p * 100,
                name="Prob. Modèle",
                marker_color=colors_bar,
                text=[f"{p*100:.1f}%" for p in model_p],
                textposition="outside",
                textfont=dict(color="#e8eaf0", size=11),
            ))
            fig_prob.add_trace(go.Scatter(
                x=names,
                y=market_p * 100,
                name="Prob. Marché",
                mode="markers+lines",
                marker=dict(color="#e05252", size=9, symbol="diamond"),
                line=dict(color="#e05252", width=1.5, dash="dot"),
            ))
            fig_prob.update_layout(
                title="Probabilités Modèle vs Marché",
                paper_bgcolor="#111318",
                plot_bgcolor="#111318",
                font=dict(color="#e8eaf0", family="DM Sans"),
                title_font=dict(color="#c9a84c", size=16, family="DM Serif Display"),
                legend=dict(bgcolor="#111318", bordercolor="#252933", borderwidth=1),
                xaxis=dict(gridcolor="#252933", tickangle=-30),
                yaxis=dict(gridcolor="#252933", title="Probabilité (%)"),
                bargap=0.3,
                height=400,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Value index chart
            fig_val = go.Figure()
            bar_colors = ["#3db87a" if v > 0 else "#e05252" for v in value_idx]
            fig_val.add_trace(go.Bar(
                x=names,
                y=value_idx * 100,
                marker_color=bar_colors,
                text=[f"{v*100:+.1f}pts" for v in value_idx],
                textposition="outside",
                textfont=dict(color="#e8eaf0", size=10),
            ))
            fig_val.add_hline(y=0, line_color="#c9a84c", line_width=1.5, line_dash="dash")
            fig_val.update_layout(
                title="Indice de Value (Modèle − Marché)",
                paper_bgcolor="#111318",
                plot_bgcolor="#111318",
                font=dict(color="#e8eaf0", family="DM Sans"),
                title_font=dict(color="#c9a84c", size=16, family="DM Serif Display"),
                xaxis=dict(gridcolor="#252933", tickangle=-30),
                yaxis=dict(gridcolor="#252933", title="Value Index (pts)"),
                bargap=0.3,
                height=350,
            )
            st.plotly_chart(fig_val, use_container_width=True)

            # Radar chart for top 5
            top5 = ranking[:min(5, n)]
            top5_order = R["order"][:min(5, n)]

            categories = ["Score Musique", "Forme Récente", "Taux Victoire", "Driver+Trainer", "Régularité"]
            fig_radar = go.Figure()
            palette = ["#c9a84c", "#4c7bc9", "#3db87a", "#e05252", "#9b59b6"]

            for ci, horse in enumerate(top5):
                orig_i = top5_order[ci]
                mm = minmax_norm
                vals = [
                    float(mm(results["music_score"])[orig_i]),
                    float(mm(np.array([h["music"]["recent_form"] for h in R["ranking"]]))[ci]),
                    float(mm(np.array([h["music"]["win_rate"] for h in R["ranking"]]))[ci]),
                    float(mm(results["human_factor"])[orig_i]),
                    float(mm(np.array([h["music"]["regularity"] for h in R["ranking"]]))[ci]),
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=f"N°{horse['numero']} {horse['nom'][:10]}",
                    line_color=palette[ci],
                    fillcolor=palette[ci],
                    opacity=0.25,
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#111318",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#252933", tickfont=dict(color="#7a8090")),
                    angularaxis=dict(gridcolor="#252933", tickcolor="#c9a84c"),
                ),
                paper_bgcolor="#111318",
                font=dict(color="#e8eaf0", family="DM Sans"),
                title=dict(text="Profil Radar — Top 5", font=dict(color="#c9a84c", size=16, family="DM Serif Display")),
                legend=dict(bgcolor="#111318", bordercolor="#252933", borderwidth=1),
                height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── TAB 3: Combinations ──────────────────────────────
        with tabs[2]:
            cot1, cot2 = st.columns(2)
            with cot1:
                st.markdown('<div class="section-title">🎯 Combinaisons Trio</div>', unsafe_allow_html=True)
                trio_data = []
                for rank_i, (combo, score) in enumerate(combos["trio"]):
                    trio_data.append({
                        "Rang": rank_i + 1,
                        "Combinaison": f"{combo[0]}–{combo[1]}–{combo[2]}",
                        "Score Composite": f"{score:.6f}",
                        "Priorité": "⭐⭐⭐" if rank_i < 3 else ("⭐⭐" if rank_i < 6 else "⭐"),
                    })
                st.dataframe(pd.DataFrame(trio_data), hide_index=True, use_container_width=True)

            with cot2:
                st.markdown('<div class="section-title">🏆 Combinaisons Quinté</div>', unsafe_allow_html=True)
                quinte_data = []
                for rank_i, (combo, score) in enumerate(combos["quinte"]):
                    quinte_data.append({
                        "Rang": rank_i + 1,
                        "Combinaison": f"{combo[0]}–{combo[1]}–{combo[2]}–{combo[3]}–{combo[4]}",
                        "Score": f"{score:.8f}",
                        "Priorité": "⭐⭐⭐" if rank_i < 3 else ("⭐⭐" if rank_i < 6 else "⭐"),
                    })
                st.dataframe(pd.DataFrame(quinte_data), hide_index=True, use_container_width=True)

        # ── TAB 4: Analysis ──────────────────────────────────
        with tabs[3]:
            st.markdown('<div class="section-title">🧠 Analyse Automatique Professionnelle</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="analysis-box">{R["analysis"]}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Detailed score table
            st.markdown('<div class="section-title">📊 Matrice des Scores Détaillés</div>', unsafe_allow_html=True)
            order = R["order"]
            detail_rows = []
            for rank_i, (orig_i, horse) in enumerate(zip(order, R["ranking"])):
                detail_rows.append({
                    "Rang":       rank_i + 1,
                    "N°":        horse["numero"],
                    "Nom":       horse["nom"],
                    "Score Musique": f"{horse['music']['score']:.3f}",
                    "Forme Récente": f"{horse['music']['recent_form']:.2f}",
                    "W% Musique":    f"{horse['music']['win_rate']*100:.0f}%",
                    "Place% Music":  f"{horse['music']['place_rate']*100:.0f}%",
                    "Régularité":    f"{horse['music']['regularity']:.3f}",
                    "Courses":       horse["music"]["n_races"],
                    "Score Compo":   f"{results['composite'][orig_i]:.4f}",
                    "Prob Modèle":   f"{R['model_prob_ranked'][rank_i]*100:.2f}%",
                    "Prob Marché":   f"{R['market_prob_ranked'][rank_i]*100:.2f}%",
                    "MC Prob":       f"{results['mc_prob'][orig_i]*100:.2f}%",
                    "Value":         f"{R['value_index_ranked'][rank_i]*100:+.2f}pts",
                    "EV":            f"{R['ev_ranked'][rank_i]:.4f}",
                })
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:3rem;margin-bottom:1rem">
<div style="text-align:center;color:#3a4055;font-size:0.78rem;font-family:'DM Mono',monospace;letter-spacing:1px">
    TURFQUANT PRO · MOTEUR QUANTITATIF PROBABILISTE · USAGE ANALYTIQUE UNIQUEMENT · ©2025
</div>
""", unsafe_allow_html=True)
