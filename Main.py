"""
QuantTurf Pro v3.2.1 - CORRECTED ENGINE
=========================================
✅ Stable probability calculation
✅ Correct place probability (top 3)
✅ Trio analysis (exact or any order)
✅ Kelly and ROI
✅ Works with identical horse data
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import zscore
from itertools import combinations, permutations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    APP_VERSION: str = "3.2.1"
    APP_NAME: str = "QuantTurf Pro"
    MC_ITERATIONS: int = 3000
    MARKET_WEIGHT: float = 0.35
    VALUE_THRESHOLD: float = 1.15
    TEMPERATURE: float = 1.5
    NOISE_BASE: float = 0.15
    KELLY_FRACTION: float = 0.25
    MIN_KELLY_ODDS: float = 2.50
    RACE_TYPES: List[str] = None
    PLACE_ODDS_FACTOR: float = 0.45
    TRIO_ANY_ORDER: bool = False

    MUSIC_POSITION_SCORES: Dict[str, float] = None
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = None
    DRAW_IMPACT_BASE: Dict[int, float] = None

    def __post_init__(self):
        if self.MUSIC_POSITION_SCORES is None:
            self.MUSIC_POSITION_SCORES = {
                "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
                "6": 2.0, "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
                "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
            }
        if self.MUSIC_RACE_TYPE_WEIGHTS is None:
            self.MUSIC_RACE_TYPE_WEIGHTS = {
                "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95,
                "s": 0.90, "c": 0.85, "x": 1.00,
            }
        if self.DRAW_IMPACT_BASE is None:
            self.DRAW_IMPACT_BASE = {
                1: 0.35, 2: 0.40, 3: 0.35, 4: 0.25, 5: 0.15,
                6: 0.05, 7: -0.05, 8: -0.12, 9: -0.18, 10: -0.24,
                11: -0.30, 12: -0.35, 13: -0.40, 14: -0.44, 15: -0.48,
                16: -0.50, 17: -0.52, 18: -0.54, 19: -0.55, 20: -0.55,
            }
        if self.RACE_TYPES is None:
            self.RACE_TYPES = ["Plat", "Attelé", "Monté", "Haies", "Steeple-chase", "Cross-country"]

CONFIG = Config()


# =============================================================================
# MUSIC PARSING (identique mais robuste)
# =============================================================================

@dataclass
class MusicMetrics:
    score: float
    regularity: float
    races_count: int
    avg_position: float
    best_position: int
    recent_form: float
    trend: float
    is_debutant: bool
    win_ratio: float
    podium_ratio: float
    win_streak: int = 0
    place_streak: int = 0
    consistency: float = 0.0


@lru_cache(maxsize=512)
def parse_music_final(music_str: str) -> MusicMetrics:
    if not music_str or music_str.strip() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0"):
        return MusicMetrics(
            score=3.0, regularity=0.50, races_count=0,
            avg_position=5.0, best_position=10, recent_form=3.0,
            trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
        )
    try:
        clean = music_str.strip().upper()
        clean = re.sub(r"[() ]", "", clean)
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        if not tokens:
            return MusicMetrics(score=3.0, regularity=0.50, races_count=0,
                avg_position=5.0, best_position=10, recent_form=3.0,
                trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0)
        raw_scores, numeric_positions = [], []
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "x"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)
        n = len(raw_scores)
        raw_scores = np.array(raw_scores)
        decay = np.array([np.exp(-0.30 * i) for i in range(n)])
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores, decay))
        recent_n = min(3, n)
        recent_decay = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores[:recent_n], recent_decay))
        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            regularity = 0.50
        if n >= 4:
            recent_avg = np.mean(raw_scores[:n // 2])
            old_avg = np.mean(raw_scores[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0
        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        consistency = 1.0 - (pos_std / 10.0 if len(numeric_positions) >= 2 else 0.5)
        consistency = max(0.0, min(1.0, consistency))
        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_count / max(n, 1),
            podium_ratio=podium_count / max(n, 1),
        )
    except Exception as e:
        logger.warning(f"Music parsing error: {str(e)}")
        return MusicMetrics(score=3.0, regularity=0.50, races_count=0,
            avg_position=5.0, best_position=10, recent_form=3.0,
            trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0)


# =============================================================================
# FEATURES & WEIGHTS (corrigés)
# =============================================================================

def draw_factor(draw: int, race_type: str, distance: int) -> float:
    if race_type != "Plat" or not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)
    base = CONFIG.DRAW_IMPACT_BASE.get(draw, -0.55)
    if distance <= 1400:
        return base * 1.60
    elif distance <= 1800:
        return base * 1.00
    else:
        return base * 0.45


def market_prob(odds: float, n_runners: int) -> float:
    if not odds or odds <= 1.01:
        return 1.0 / max(n_runners, 2)
    return 1.0 / float(odds)


def get_weights_final(race_type: str) -> Dict[str, float]:
    if race_type == "Plat":
        return {
            "horse_music_score": 0.18, "horse_recent_form": 0.10, "horse_regularity": 0.04,
            "horse_trend": 0.02, "horse_win_ratio": 0.01,
            "driver_music_score": 0.17, "driver_recent_form": 0.10, "driver_regularity": 0.04,
            "driver_trend": 0.01, "driver_win_ratio": 0.01,
            "trainer_music_score": 0.13, "trainer_recent_form": 0.08, "trainer_regularity": 0.04,
            "trainer_trend": 0.01, "trainer_win_ratio": 0.01,
            "draw_factor": 0.03, "synergy_score": 0.02,
        }
    elif race_type in ("Attelé", "Monté"):
        return {
            "horse_music_score": 0.16, "horse_recent_form": 0.08, "horse_regularity": 0.03,
            "horse_trend": 0.02, "horse_win_ratio": 0.01,
            "driver_music_score": 0.21, "driver_recent_form": 0.12, "driver_regularity": 0.04,
            "driver_trend": 0.02, "driver_win_ratio": 0.01,
            "trainer_music_score": 0.12, "trainer_recent_form": 0.07, "trainer_regularity": 0.03,
            "trainer_trend": 0.01, "trainer_win_ratio": 0.01,
            "draw_factor": 0.00, "synergy_score": 0.03,
        }
    else:
        return {
            "horse_music_score": 0.20, "horse_recent_form": 0.10, "horse_regularity": 0.05,
            "horse_trend": 0.02, "horse_win_ratio": 0.01,
            "driver_music_score": 0.14, "driver_recent_form": 0.08, "driver_regularity": 0.03,
            "driver_trend": 0.02, "driver_win_ratio": 0.01,
            "trainer_music_score": 0.16, "trainer_recent_form": 0.09, "trainer_regularity": 0.04,
            "trainer_trend": 0.01, "trainer_win_ratio": 0.01,
            "draw_factor": 0.00, "synergy_score": 0.02,
        }


def composite_score_final(feat: Dict, weights: Dict) -> float:
    """Score composite robuste avec clipping"""
    score = 0.0
    # Horse
    score += weights.get("horse_music_score", 0.18) * np.clip(feat.get("horse_music_score", 3.0), 0, 12)
    score += weights.get("horse_recent_form", 0.10) * np.clip(feat.get("horse_recent_form", 3.0), 0, 12)
    score += weights.get("horse_regularity", 0.04) * np.clip(feat.get("horse_regularity", 0.5), 0, 1) * 10.0
    score += weights.get("horse_trend", 0.02) * (np.clip(feat.get("horse_trend", 0.0), -1, 1) + 1.0) * 5.0
    score += weights.get("horse_win_ratio", 0.01) * np.clip(feat.get("horse_win_ratio", 0.0), 0, 1) * 20.0
    # Driver
    score += weights.get("driver_music_score", 0.17) * np.clip(feat.get("driver_music_score", 3.0), 0, 12)
    score += weights.get("driver_recent_form", 0.10) * np.clip(feat.get("driver_recent_form", 3.0), 0, 12)
    score += weights.get("driver_regularity", 0.04) * np.clip(feat.get("driver_regularity", 0.5), 0, 1) * 10.0
    score += weights.get("driver_trend", 0.01) * (np.clip(feat.get("driver_trend", 0.0), -1, 1) + 1.0) * 5.0
    score += weights.get("driver_win_ratio", 0.01) * np.clip(feat.get("driver_win_ratio", 0.0), 0, 1) * 20.0
    # Trainer
    score += weights.get("trainer_music_score", 0.13) * np.clip(feat.get("trainer_music_score", 3.0), 0, 12)
    score += weights.get("trainer_recent_form", 0.08) * np.clip(feat.get("trainer_recent_form", 3.0), 0, 12)
    score += weights.get("trainer_regularity", 0.04) * np.clip(feat.get("trainer_regularity", 0.5), 0, 1) * 10.0
    score += weights.get("trainer_trend", 0.01) * (np.clip(feat.get("trainer_trend", 0.0), -1, 1) + 1.0) * 5.0
    score += weights.get("trainer_win_ratio", 0.01) * np.clip(feat.get("trainer_win_ratio", 0.0), 0, 1) * 20.0
    # Draw
    if weights.get("draw_factor", 0) > 0:
        score += weights["draw_factor"] * (np.clip(feat.get("draw_factor", 0.0), -1, 1) + 1.0) * 5.0
    # Synergy
    horse_m = np.clip(feat.get("horse_music_score", 3.0), 0, 12)
    driver_m = np.clip(feat.get("driver_music_score", 3.0), 0, 12)
    trainer_m = np.clip(feat.get("trainer_music_score", 3.0), 0, 12)
    synergy = min(horse_m, driver_m, trainer_m) / (max(horse_m, driver_m, trainer_m) + 1e-9)
    score += weights.get("synergy_score", 0.02) * synergy * 10.0
    return max(0.01, score)


def softmax(scores: np.ndarray, temperature: float = CONFIG.TEMPERATURE) -> np.ndarray:
    s = np.array(scores, dtype=float) / max(temperature, 0.1)
    # Écrêtage pour éviter l'explosion exponentielle
    s = np.clip(s, -50, 50)
    s -= s.max()
    e = np.exp(s)
    probs = e / (e.sum() + 1e-9)
    # Si une probabilité dépasse 0.99 et que n>2, on répartit
    if len(probs) > 2 and np.max(probs) > 0.99:
        probs = np.ones_like(probs) / len(probs)
    return probs


def logit_calibration(raw_probs: np.ndarray) -> np.ndarray:
    eps = 1e-9
    raw_probs = np.clip(raw_probs, eps, 1 - eps)
    logit = np.log(raw_probs / (1 - raw_probs))
    logit = logit - logit.mean() * 0.1
    calibrated = 1.0 / (1.0 + np.exp(-logit))
    return calibrated / calibrated.sum()


def bayesian_blend(model_probs: np.ndarray, market_probs: np.ndarray, market_weight: float) -> np.ndarray:
    mp = np.array(market_probs, dtype=float)
    if mp.sum() < 1e-9:
        mp = np.ones(len(model_probs)) / len(model_probs)
    else:
        mp /= mp.sum()
    eps = 1e-9
    lo_model = np.log((model_probs + eps) / (1 - model_probs + eps))
    lo_market = np.log((mp + eps) / (1 - mp + eps))
    lo_blend = (1 - market_weight) * lo_model + market_weight * lo_market
    blended = 1.0 / (1.0 + np.exp(-lo_blend))
    return blended / blended.sum()


# =============================================================================
# MONTE CARLO (corrigé)
# =============================================================================

def monte_carlo_final(features_list: List[Dict], weights: Dict, n_iter: int = CONFIG.MC_ITERATIONS) -> Dict:
    n = len(features_list)
    win_counts = np.zeros(n)
    place_counts = np.zeros(n)
    finishing_orders = np.zeros((n_iter, n), dtype=int)
    
    # Scores de base
    base_scores = np.array([composite_score_final(f, weights) for f in features_list])
    # Si tous les scores sont égaux, on ajoute un petit bruit pour éviter l'indétermination
    if np.std(base_scores) < 1e-6:
        base_scores += np.random.normal(0, 0.01, n)
    
    # Facteurs de bruit par cheval (dépendent de la régularité)
    noise_factors = np.array([
        2.20 if f.get("horse_is_debutant", False) else
        1.60 if f.get("horse_regularity", 0.5) < 0.30 else
        0.70 if f.get("horse_regularity", 0.5) > 0.80 else 1.00
        for f in features_list
    ])
    
    for it in range(n_iter):
        noises = np.random.normal(0, CONFIG.NOISE_BASE * noise_factors, n)
        noisy = base_scores * np.exp(noises)
        noisy = np.maximum(noisy, 0.001)
        probs = softmax(noisy, temperature=CONFIG.TEMPERATURE)
        # Ordre d'arrivée (ordre décroissant de proba)
        order = np.argsort(-probs)
        finishing_orders[it] = order
        # Gagnant
        winner = order[0]
        win_counts[winner] += 1
        # Places (top 3)
        for p in order[:3]:
            place_counts[p] += 1
    
    win_probs = win_counts / n_iter
    place_probs = place_counts / n_iter
    # Calcul des moyennes et écarts-types des probabilités simulées
    # On recrée les probs pour chaque itération (coûteux mais nécessaire pour std)
    all_probs = np.zeros((n_iter, n))
    for it in range(n_iter):
        noises = np.random.normal(0, CONFIG.NOISE_BASE * noise_factors, n)
        noisy = base_scores * np.exp(noises)
        noisy = np.maximum(noisy, 0.001)
        all_probs[it] = softmax(noisy, temperature=CONFIG.TEMPERATURE)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    vol_per_horse = std_probs / (mean_probs + 1e-9)
    
    return {
        "simulated_probs": win_probs,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "vol_per_horse": vol_per_horse,
        "place_probs": place_probs,
        "finishing_orders": finishing_orders,
    }


# =============================================================================
# KELLY & ROI
# =============================================================================

def calculate_kelly_bet(prob: float, odds: float, kelly_fraction: float = CONFIG.KELLY_FRACTION) -> Tuple[float, float]:
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.05:
        return 0.0, 0.0
    q = 1.0 - prob
    b = odds - 1.0
    if b <= 0:
        return 0.0, 0.0
    kelly = (prob * b - q) / b
    kelly = max(0.0, kelly)
    fractional_kelly = kelly * kelly_fraction
    return float(kelly), float(fractional_kelly)


def calculate_roi(prob: float, odds: float, bet_amount: float = 100.0) -> float:
    if bet_amount <= 0 or odds <= 1.0:
        return 0.0
    expected_winnings = bet_amount * odds * prob
    expected_loss = bet_amount * (1 - prob)
    expected_value = expected_winnings - expected_loss
    return (expected_value / bet_amount) * 100.0


# =============================================================================
# TRIO & PLACE
# =============================================================================

def analyze_trios(results: List[Dict], finishing_orders: np.ndarray) -> List[Dict]:
    n_horses = len(results)
    n_iter = finishing_orders.shape[0]
    if n_horses < 3:
        return []
    
    if CONFIG.TRIO_ANY_ORDER:
        combos = list(combinations(range(n_horses), 3))
        combo_counts = {c: 0 for c in combos}
        for it in range(n_iter):
            top3 = tuple(sorted(finishing_orders[it][:3]))
            if top3 in combo_counts:
                combo_counts[top3] += 1
        trio_stats = []
        for combo, count in combo_counts.items():
            prob = count / n_iter
            if prob < 0.002:
                continue
            i1, i2, i3 = combo
            p1 = results[i1]["model_prob"] / 100
            p2 = results[i2]["model_prob"] / 100
            p3 = results[i3]["model_prob"] / 100
            est_prob = 6.0 * p1 * p2 * p3
            est_odds = 1.0 / max(est_prob, 0.001)
            est_odds = np.clip(est_odds, 5.0, 100.0)
            roi = calculate_roi(prob, est_odds, 10)
            trio_stats.append({
                "rank": 0,
                "numbers": (results[i1]["number"], results[i2]["number"], results[i3]["number"]),
                "names": (results[i1]["name"][:10], results[i2]["name"][:10], results[i3]["name"][:10]),
                "prob_pct": round(prob * 100, 2),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(roi, 1),
                "p1": round(p1 * 100, 1),
                "p2": round(p2 * 100, 1),
                "p3": round(p3 * 100, 1),
            })
    else:
        perms = list(permutations(range(n_horses), 3))
        perm_counts = {p: 0 for p in perms}
        for it in range(n_iter):
            top3 = tuple(finishing_orders[it][:3])
            if top3 in perm_counts:
                perm_counts[top3] += 1
        trio_stats = []
        for perm, count in perm_counts.items():
            prob = count / n_iter
            if prob < 0.001:
                continue
            i1, i2, i3 = perm
            p1 = results[i1]["model_prob"] / 100
            p2 = results[i2]["model_prob"] / 100
            p3 = results[i3]["model_prob"] / 100
            est_prob = p1 * p2 * p3
            est_odds = 1.0 / max(est_prob, 0.001)
            est_odds = np.clip(est_odds, 5.0, 100.0)
            roi = calculate_roi(prob, est_odds, 10)
            trio_stats.append({
                "rank": 0,
                "numbers": (results[i1]["number"], results[i2]["number"], results[i3]["number"]),
                "names": (results[i1]["name"][:10], results[i2]["name"][:10], results[i3]["name"][:10]),
                "prob_pct": round(prob * 100, 2),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(roi, 1),
                "p1": round(p1 * 100, 1),
                "p2": round(p2 * 100, 1),
                "p3": round(p3 * 100, 1),
            })
    trio_stats.sort(key=lambda x: x["expected_roi"], reverse=True)
    for i, t in enumerate(trio_stats[:10]):
        t["rank"] = i + 1
    return trio_stats[:10]


def find_best_place_bet(results: List[Dict]) -> Dict:
    best = None
    best_roi = -999
    for r in results:
        place_prob = r["place_prob"] / 100
        if place_prob < 0.10:
            continue
        win_odds = r["odds"]
        if win_odds <= 2.0:
            factor = 0.50
        elif win_odds <= 5.0:
            factor = 0.45
        elif win_odds <= 10.0:
            factor = 0.40
        else:
            factor = 0.35
        place_odds = max(1.5, win_odds * factor)
        roi = calculate_roi(place_prob, place_odds, 100)
        if roi > best_roi:
            best_roi = roi
            kelly, kelly_frac = calculate_kelly_bet(place_prob, place_odds)
            best = {
                "number": r["number"],
                "name": r["name"],
                "win_prob": r["model_prob"],
                "place_prob": r["place_prob"],
                "estimated_place_odds": round(place_odds, 2),
                "expected_roi_place": round(roi, 1),
                "kelly_criterion": round(kelly, 4),
                "kelly_bet_fraction": round(kelly_frac, 4),
            }
    return best


# =============================================================================
# MOTEUR PRINCIPAL (corrigé)
# =============================================================================

def run_engine_final(race_info: Dict, horses: List[Dict], mc_iter: int = CONFIG.MC_ITERATIONS, market_weight: float = CONFIG.MARKET_WEIGHT, value_threshold: float = CONFIG.VALUE_THRESHOLD) -> Dict:
    start_time = time.time()
    try:
        n_runners = len(horses)
        race_type = race_info.get("race_type", "Plat")
        distance = int(race_info.get("distance", 1600))
        
        # Construction des features
        feats = []
        for h in horses:
            horse_music = parse_music_final(h.get("horse_music", ""))
            driver_music = parse_music_final(h.get("driver_music", ""))
            trainer_music = parse_music_final(h.get("trainer_music", ""))
            feat = {
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                "horse_music_score": horse_music.score,
                "horse_recent_form": horse_music.recent_form,
                "horse_regularity": horse_music.regularity,
                "horse_trend": horse_music.trend,
                "horse_win_ratio": horse_music.win_ratio,
                "horse_is_debutant": horse_music.is_debutant,
                "driver_music_score": driver_music.score,
                "driver_recent_form": driver_music.recent_form,
                "driver_regularity": driver_music.regularity,
                "driver_trend": driver_music.trend,
                "driver_win_ratio": driver_music.win_ratio,
                "trainer_music_score": trainer_music.score,
                "trainer_recent_form": trainer_music.recent_form,
                "trainer_regularity": trainer_music.regularity,
                "trainer_trend": trainer_music.trend,
                "trainer_win_ratio": trainer_music.win_ratio,
                "draw_factor": draw_factor(h.get("draw", 0), race_type, distance),
                "market_prob": market_prob(h.get("odds", 0), n_runners),
            }
            feats.append(feat)
        
        weights = get_weights_final(race_type)
        
        # Scores composites
        scores = np.array([composite_score_final(f, weights) for f in feats])
        # Vérification : si écart-type nul, ajout de bruit
        if np.std(scores) < 1e-6:
            scores += np.random.normal(0, 0.01, n_runners)
        
        # Probabilités par softmax
        sm_probs = softmax(scores)
        cal_probs = logit_calibration(sm_probs)
        
        # Probabilités de marché
        raw_mkt = np.array([f["market_prob"] for f in feats])
        if raw_mkt.sum() < 1e-9:
            raw_mkt = np.ones(n_runners) / n_runners
        norm_mkt = raw_mkt / raw_mkt.sum()
        
        has_odds = any(h.get("odds", 0) > CONFIG.MIN_KELLY_ODDS for h in horses)
        if has_odds:
            bayes_probs = bayesian_blend(cal_probs, norm_mkt, market_weight)
        else:
            bayes_probs = cal_probs
        
        # Monte Carlo
        mc = monte_carlo_final(feats, weights, n_iter=mc_iter)
        
        # Fusion finale
        final_probs = 0.55 * bayes_probs + 0.45 * mc["mean_probs"]
        final_probs /= final_probs.sum()
        
        # Construire les résultats
        results = []
        for i, (feat, horse) in enumerate(zip(feats, horses)):
            ratio = final_probs[i] / (norm_mkt[i] + 1e-9)
            is_value = ratio >= value_threshold and final_probs[i] >= 0.04
            kelly, kelly_frac = calculate_kelly_bet(final_probs[i], horse.get("odds", 2.0))
            roi = calculate_roi(final_probs[i], horse.get("odds", 2.0), 100.0)
            results.append({
                "rank": 0,
                "number": horse.get("number", i+1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "model_prob": round(float(final_probs[i]) * 100, 2),
                "market_prob": round(float(norm_mkt[i]) * 100, 2),
                "place_prob": round(float(mc["place_probs"][i]) * 100, 2),
                "composite_score": round(float(scores[i]), 4),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": is_value,
                "kelly_criterion": round(kelly, 4),
                "kelly_bet_fraction": round(kelly_frac, 4),
                "expected_roi": round(roi, 2),
            })
        
        # Tri par probabilité
        results.sort(key=lambda x: x["model_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        # Trios et place
        trios = analyze_trios(results, mc["finishing_orders"])
        best_place = find_best_place_bet(results)
        
        # Indices
        sorted_probs = sorted([r["model_prob"] for r in results], reverse=True)
        if len(sorted_probs) >= 2:
            gap = sorted_probs[0] - sorted_probs[1]
            conf_idx = min(100.0, round(45.0 + gap * 2.2, 1))
        else:
            conf_idx = 50.0
        vol_idx = min(100.0, round(mc["vol_per_horse"].mean() * 55.0, 1))
        
        if has_odds:
            raw_overround = sum(1.0 / h["odds"] for h in horses if h.get("odds", 0) > 1.01)
            overround_pct = round((raw_overround - 1.0) * 100, 1)
        else:
            overround_pct = None
        
        return {
            "results": results,
            "trios": trios,
            "best_place": best_place,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "execution_time": round(time.time() - start_time, 2),
        }
    except Exception as e:
        logger.error(f"Engine error: {str(e)}")
        raise


# =============================================================================
# INTERFACE STREAMLIT (simplifiée mais complète)
# =============================================================================

def apply_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #07071a 0%, #0d1b2a 40%, #12192b 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a, #07071a); }
    h1, h2, h3 { color: #e8e8e8 !important; }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown(f"""
    <div style="text-align:center; padding: 22px 0;">
        <h1 style="font-size:2.8em; background: linear-gradient(90deg,#00ff88,#00b4d8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}
        </h1>
        <p style="color:#6b7fa3;">Moteur corrigé - Probabilités stables</p>
    </div>
    """, unsafe_allow_html=True)

def init_session_state():
    if "horses_data" not in st.session_state:
        st.session_state.horses_data = pd.DataFrame({
            "N°": range(1, 11),
            "Nom": [f"Cheval {i+1}" for i in range(10)],
            "Cote": [5.0] * 10,
            "Musique Cheval": [""] * 10,
            "Musique Driver": [""] * 10,
            "Musique Entraîneur": [""] * 10,
            "Corde": [0] * 10,
        })
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

def main():
    st.set_page_config(page_title=f"🏇 {CONFIG.APP_NAME}", layout="wide")
    init_session_state()
    apply_css()
    render_header()
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        mc_iter = st.slider("MC Itérations", 500, 5000, CONFIG.MC_ITERATIONS, 250)
        mw = st.slider("Poids Marché", 0.0, 0.60, CONFIG.MARKET_WEIGHT, 0.05)
        vt = st.slider("Seuil Value", 1.05, 1.60, CONFIG.VALUE_THRESHOLD, 0.05)
        trio_any = st.checkbox("Trio désordre", value=CONFIG.TRIO_ANY_ORDER)
        CONFIG.TRIO_ANY_ORDER = trio_any
    
    tab1, tab2 = st.tabs(["📥 Données", "📊 Résultats"])
    
    with tab1:
        st.markdown("## 🏁 Course")
        c1, c2, c3 = st.columns(3)
        with c1:
            race_type = st.selectbox("Type", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            discipline = st.text_input("Prix")
        
        st.markdown("---\n## 🐎 Données Chevaux")
        edited_df = st.data_editor(
            st.session_state.horses_data,
            use_container_width=True,
            num_rows="dynamic",
            height=400,
        )
        if edited_df is not None:
            st.session_state.horses_data = edited_df
        
        if st.button("🚀 ANALYSER", use_container_width=True):
            horses_list = []
            for _, row in st.session_state.horses_data.iterrows():
                try:
                    horses_list.append({
                        "number": int(row["N°"]),
                        "name": str(row["Nom"]),
                        "odds": float(row["Cote"]),
                        "horse_music": str(row["Musique Cheval"]),
                        "driver_music": str(row["Musique Driver"]),
                        "trainer_music": str(row["Musique Entraîneur"]),
                        "draw": int(row["Corde"]),
                    })
                except:
                    st.error(f"Erreur ligne {_}")
                    return
            with st.spinner("Calcul en cours..."):
                pred = run_engine_final(
                    {"race_type": race_type, "distance": distance, "discipline": discipline},
                    horses_list,
                    mc_iter=mc_iter, market_weight=mw, value_threshold=vt
                )
                st.session_state.prediction = pred
                st.success(f"Terminé en {pred['execution_time']}s")
    
    with tab2:
        if st.session_state.prediction is None:
            st.info("Lancez l'analyse d'abord")
        else:
            pred = st.session_state.prediction
            st.markdown("## 📊 Classement")
            df_res = pd.DataFrame([{
                "Rg": r["rank"],
                "N°": r["number"],
                "Nom": r["name"],
                "Gagnant%": f"{r['model_prob']:.1f}",
                "Placé%": f"{r['place_prob']:.1f}",
                "Kelly%": f"{r['kelly_bet_fraction']*100:.2f}",
                "ROI%": f"{r['expected_roi']:.1f}",
                "Value": "🟢" if r["is_value_bet"] else "⚪"
            } for r in pred["results"]])
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            
            if pred["best_place"]:
                st.markdown("---\n## 🎯 Meilleur cheval pour le PLACÉ")
                bp = pred["best_place"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("N°", bp["number"])
                col2.metric("Nom", bp["name"])
                col3.metric("Prob. Placé", f"{bp['place_prob']:.1f}%")
                col4.metric("ROI Placé", f"{bp['expected_roi_place']:.1f}%")
                st.markdown(f"**Kelly recommandé (25%)** : {bp['kelly_bet_fraction']:.2%} du bankroll")
            
            st.markdown("---\n## 🎲 Top 10 Trios")
            if pred["trios"]:
                df_trio = pd.DataFrame([{
                    "Rg": t["rank"],
                    "Trio": f"{t['numbers'][0]}-{t['numbers'][1]}-{t['numbers'][2]}",
                    "Prob%": t["prob_pct"],
                    "Cote est.": t["estimated_odds"],
                    "ROI%": t["expected_roi"]
                } for t in pred["trios"]])
                st.dataframe(df_trio, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun trio significatif")

if __name__ == "__main__":
    main()
