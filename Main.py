"""
QuantTurf Pro v3.1.1.3 ENHANCED
================================
✅ 10 Best Trio Combinations (with ROI)
✅ Profitable Place Bet Recommendation
✅ Kelly Sizing for Each Bet Type
✅ Expected ROI for All Recommendations

Version: 3.1.1.3 (Enhanced Betting)
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
# CONFIG (Same as v3.1.1.2)
# =============================================================================

@dataclass
class Config:
    APP_VERSION: str = "3.1.1.3"
    APP_NAME: str = "QuantTurf Pro"
    MC_ITERATIONS: int = 3000
    MARKET_WEIGHT: float = 0.35
    VALUE_THRESHOLD: float = 1.15
    TEMPERATURE: float = 1.5
    NOISE_BASE: float = 0.15
    KELLY_FRACTION: float = 0.25
    MIN_KELLY_ODDS: float = 2.50
    RACE_TYPES: List[str] = None
    
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
# MUSIC PARSING (Same as v3.1.1.2)
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
    """Parse music history"""
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
            return MusicMetrics(
                score=3.0, regularity=0.50, races_count=0,
                avg_position=5.0, best_position=10, recent_form=3.0,
                trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
            )
        
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
        return MusicMetrics(
            score=3.0, regularity=0.50, races_count=0,
            avg_position=5.0, best_position=10, recent_form=3.0,
            trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
        )


# =============================================================================
# FEATURES & WEIGHTS (Same as v3.1.1.2)
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
    """Race-specific weights"""
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
    """Composite score calculation"""
    score = 0.0
    
    score += weights.get("horse_music_score", 0.18) * feat.get("horse_music_score", 3.0)
    score += weights.get("horse_recent_form", 0.10) * feat.get("horse_recent_form", 3.0)
    score += weights.get("horse_regularity", 0.04) * feat.get("horse_regularity", 0.5) * 10.0
    score += weights.get("horse_trend", 0.02) * (feat.get("horse_trend", 0.0) + 1.0) * 5.0
    score += weights.get("horse_win_ratio", 0.01) * feat.get("horse_win_ratio", 0.0) * 20.0
    
    score += weights.get("driver_music_score", 0.17) * feat.get("driver_music_score", 3.0)
    score += weights.get("driver_recent_form", 0.10) * feat.get("driver_recent_form", 3.0)
    score += weights.get("driver_regularity", 0.04) * feat.get("driver_regularity", 0.5) * 10.0
    score += weights.get("driver_trend", 0.01) * (feat.get("driver_trend", 0.0) + 1.0) * 5.0
    score += weights.get("driver_win_ratio", 0.01) * feat.get("driver_win_ratio", 0.0) * 20.0
    
    score += weights.get("trainer_music_score", 0.13) * feat.get("trainer_music_score", 3.0)
    score += weights.get("trainer_recent_form", 0.08) * feat.get("trainer_recent_form", 3.0)
    score += weights.get("trainer_regularity", 0.04) * feat.get("trainer_regularity", 0.5) * 10.0
    score += weights.get("trainer_trend", 0.01) * (feat.get("trainer_trend", 0.0) + 1.0) * 5.0
    score += weights.get("trainer_win_ratio", 0.01) * feat.get("trainer_win_ratio", 0.0) * 20.0
    
    if weights.get("draw_factor", 0) > 0:
        score += weights["draw_factor"] * (feat.get("draw_factor", 0.0) + 1.0) * 5.0
    
    horse_m = feat.get("horse_music_score", 3.0)
    driver_m = feat.get("driver_music_score", 3.0)
    trainer_m = feat.get("trainer_music_score", 3.0)
    all_scores = [horse_m, driver_m, trainer_m]
    synergy = min(all_scores) / (max(all_scores) + 1e-9)
    score += weights.get("synergy_score", 0.02) * synergy * 10.0
    
    return max(0.01, score)


def softmax(scores: np.ndarray, temperature: float = CONFIG.TEMPERATURE) -> np.ndarray:
    s = np.array(scores, dtype=float) / temperature
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()


def logit_calibration(raw_probs: np.ndarray) -> np.ndarray:
    eps = 1e-9
    logit = np.log((raw_probs + eps) / (1 - raw_probs + eps))
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


def monte_carlo_final(features_list: List[Dict], weights: Dict, n_iter: int = CONFIG.MC_ITERATIONS) -> Dict:
    """Monte Carlo simulation"""
    n = len(features_list)
    all_probs = np.zeros((n_iter, n))
    win_counts = np.zeros(n)
    
    base_scores = np.array([composite_score_final(f, weights) for f in features_list])
    noise_factors = np.array([2.20 if f.get("horse_is_debutant", False) else 1.60 if f.get("horse_regularity", 0.5) < 0.30 else 0.70 if f.get("horse_regularity", 0.5) > 0.80 else 1.00 for f in features_list])
    
    for it in range(n_iter):
        noises = np.random.normal(0, CONFIG.NOISE_BASE * noise_factors, n)
        noisy = base_scores * np.exp(noises)
        noisy = np.maximum(noisy, 0.001)
        probs = softmax(noisy)
        all_probs[it] = probs
        winner = np.random.choice(n, p=probs)
        win_counts[winner] += 1
    
    simulated_probs = win_counts / n_iter
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    vol_per_horse = std_probs / (mean_probs + 1e-9)
    
    place_counts = np.zeros(n)
    for it in range(n_iter):
        top2 = np.argsort(-all_probs[it])[:2]
        place_counts[top2] += 1
    place_probs = place_counts / n_iter
    
    return {
        "simulated_probs": simulated_probs,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "vol_per_horse": vol_per_horse,
        "place_probs": place_probs,
        "all_probs": all_probs,  # NEW: Return all probabilities for trio analysis
    }


def calculate_kelly_bet(prob: float, odds: float, kelly_fraction: float = CONFIG.KELLY_FRACTION) -> Tuple[float, float]:
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.10:
        return 0.0, 0.0
    q = 1.0 - prob
    b = odds - 1.0
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
# NEW: TRIO & PLACE ANALYSIS
# =============================================================================

def analyze_trios(results: List[Dict], mc_data: Dict, all_mc_probs: np.ndarray) -> List[Dict]:
    """
    Analyze all possible Trio combinations with probabilities
    Returns top 10 by expected ROI
    """
    n = len(results)
    
    # Get all horses sorted by rank
    sorted_results = sorted(results, key=lambda x: x["rank"])
    
    # Generate all permutations of 3 horses (for Trio - order matters)
    trio_perms = list(permutations(range(n), 3))
    
    trio_stats = []
    
    # For each trio permutation
    for perm in trio_perms:
        i1, i2, i3 = perm
        h1, h2, h3 = sorted_results[i1], sorted_results[i2], sorted_results[i3]
        
        # Probability: P(horse1 wins AND horse2 second AND horse3 third)
        # Using MC data: average probability across iterations
        trio_wins = 0
        for it in range(len(all_mc_probs)):
            # Get top 3 in this iteration
            top3 = np.argsort(-all_mc_probs[it])[:3]
            if list(top3) == [i1, i2, i3]:
                trio_wins += 1
        
        prob_trio = trio_wins / len(all_mc_probs) if len(all_mc_probs) > 0 else 0
        
        # Estimate Trio odds (very rough: product of individual odds / correlation)
        # For simplicity: assume 1st wins at his prob, then 2nd from remaining, then 3rd from remaining
        p1 = h1["model_prob"] / 100
        p2 = h2["model_prob"] / 100
        p3 = h3["model_prob"] / 100
        
        # Approximate combo prob (assuming some correlation)
        combo_prob = p1 * p2 * p3 / (0.01 ** 2)  # Rough normalization
        combo_prob = min(combo_prob, 0.15)  # Cap at reasonable value
        
        # Estimate odds for Trio (very rough)
        trio_odds_estimate = 1 / (prob_trio + 1e-9) if prob_trio > 0 else 50
        trio_odds_estimate = min(100, max(5, trio_odds_estimate))
        
        roi = calculate_roi(prob_trio, trio_odds_estimate, 10)
        
        trio_stats.append({
            "rank": len(trio_stats) + 1,
            "numbers": (h1["number"], h2["number"], h3["number"]),
            "names": (h1["name"][:10], h2["name"][:10], h3["name"][:10]),
            "prob_pct": round(prob_trio * 100, 2),
            "estimated_odds": round(trio_odds_estimate, 1),
            "expected_roi": round(roi, 1),
            "p1": round(p1 * 100, 1),
            "p2": round(p2 * 100, 1),
            "p3": round(p3 * 100, 1),
        })
    
    # Sort by ROI descending
    trio_stats.sort(key=lambda x: x["expected_roi"], reverse=True)
    
    # Return top 10
    return trio_stats[:10]


def find_best_place_bet(results: List[Dict]) -> Dict:
    """
    Find best horse to bet on for place (top 3)
    Optimize for ROI considering odds and place probability
    """
    best_place_bet = None
    best_roi = -999
    
    for r in results:
        # Place probability is higher than win
        place_prob = r["place_prob"] / 100
        
        # Estimate place odds (place odds typically 30-50% of win odds)
        # Formula: place_odds ≈ win_odds * 0.40
        if r["odds"] > 0:
            place_odds_estimate = max(1.5, r["odds"] * 0.40)
        else:
            place_odds_estimate = 2.0
        
        # Calculate ROI for place bet
        roi_place = calculate_roi(place_prob, place_odds_estimate, 100)
        
        if roi_place > best_roi and place_prob > 0.10:  # Min 10% place prob
            best_roi = roi_place
            best_place_bet = {
                "number": r["number"],
                "name": r["name"],
                "win_prob": r["model_prob"],
                "place_prob": r["place_prob"],
                "estimated_place_odds": round(place_odds_estimate, 2),
                "expected_roi_place": round(roi_place, 1),
                "kelly_fraction": round((best_place_bet or {}).get("kelly_fraction", 0), 4) if best_place_bet else 0,
            }
    
    # Calculate Kelly for place bet
    if best_place_bet:
        place_prob = best_place_bet["place_prob"] / 100
        kelly, kelly_frac = calculate_kelly_bet(place_prob, best_place_bet["estimated_place_odds"])
        best_place_bet["kelly_criterion"] = round(kelly, 4)
        best_place_bet["kelly_bet_fraction"] = round(kelly_frac, 4)
    
    return best_place_bet


# =============================================================================
# MAIN ENGINE (Same as v3.1.1.2 with new return values)
# =============================================================================

def run_engine_final(race_info: Dict, horses: List[Dict], mc_iter: int = CONFIG.MC_ITERATIONS, market_weight: float = CONFIG.MARKET_WEIGHT, value_threshold: float = CONFIG.VALUE_THRESHOLD) -> Dict:
    """Main prediction engine with Trio & Place analysis"""
    start_time = time.time()
    
    try:
        n_runners = len(horses)
        race_info["n_runners"] = n_runners
        race_type = race_info.get("race_type", "Plat")
        distance = int(race_info.get("distance", 1600))
        
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
                "horse_races_count": horse_music.races_count,
                "horse_is_debutant": horse_music.is_debutant,
                "driver_music_score": driver_music.score,
                "driver_recent_form": driver_music.recent_form,
                "driver_regularity": driver_music.regularity,
                "driver_trend": driver_music.trend,
                "driver_win_ratio": driver_music.win_ratio,
                "driver_races_count": driver_music.races_count,
                "trainer_music_score": trainer_music.score,
                "trainer_recent_form": trainer_music.recent_form,
                "trainer_regularity": trainer_music.regularity,
                "trainer_trend": trainer_music.trend,
                "trainer_win_ratio": trainer_music.win_ratio,
                "trainer_races_count": trainer_music.races_count,
                "draw_factor": draw_factor(h.get("draw", 0), race_type, distance),
                "market_prob": market_prob(h.get("odds", 0), n_runners),
            }
            feats.append(feat)
        
        df = pd.DataFrame(feats)
        norm_cols = ["horse_music_score", "horse_recent_form", "horse_regularity", "driver_music_score", "driver_recent_form", "driver_regularity", "trainer_music_score", "trainer_recent_form", "trainer_regularity"]
        
        for col in norm_cols:
            if col in df.columns:
                vals = df[col].values.astype(float)
                std = vals.std()
                if std > 1e-9:
                    df[f"{col}_z"] = (vals - vals.mean()) / std
                else:
                    df[f"{col}_z"] = 0.0
        
        feats = df.to_dict("records")
        
        weights = get_weights_final(race_type)
        scores = np.array([composite_score_final(f, weights) for f in feats])
        
        sm_probs = softmax(scores)
        cal_probs = logit_calibration(sm_probs)
        
        raw_mkt = np.array([f["market_prob"] for f in feats])
        if raw_mkt.sum() < 1e-9:
            raw_mkt = np.ones(n_runners) / n_runners
        norm_mkt = raw_mkt / raw_mkt.sum()
        
        has_odds = any(h.get("odds", 0) > CONFIG.MIN_KELLY_ODDS for h in horses)
        if has_odds:
            bayes_probs = bayesian_blend(cal_probs, norm_mkt, CONFIG.MARKET_WEIGHT)
        else:
            bayes_probs = cal_probs
        
        mc = monte_carlo_final(feats, weights, n_iter=mc_iter)
        
        final_probs = 0.55 * bayes_probs + 0.45 * mc["mean_probs"]
        final_probs /= final_probs.sum()
        
        prob_z = zscore(final_probs)
        
        results = []
        for i, (feat, horse) in enumerate(zip(feats, horses)):
            ratio = final_probs[i] / (norm_mkt[i] + 1e-9)
            is_value = ratio >= value_threshold and final_probs[i] >= 0.04
            
            kelly, kelly_frac = calculate_kelly_bet(final_probs[i], horse.get("odds", 2.0))
            roi = calculate_roi(final_probs[i], horse.get("odds", 2.0), 100.0)
            
            result = {
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "model_prob": round(float(final_probs[i]) * 100, 2),
                "market_prob": round(float(norm_mkt[i]) * 100, 2),
                "place_prob": round(float(mc["place_probs"][i]) * 100, 2),
                "composite_score": round(float(scores[i]), 4),
                "horse_music": round(feat.get("horse_music_score", 0.0), 2),
                "horse_form": round(feat.get("horse_recent_form", 0.0), 2),
                "driver_music": round(feat.get("driver_music_score", 0.0), 2),
                "driver_form": round(feat.get("driver_recent_form", 0.0), 2),
                "trainer_music": round(feat.get("trainer_music_score", 0.0), 2),
                "trainer_form": round(feat.get("trainer_recent_form", 0.0), 2),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": is_value,
                "kelly_criterion": round(kelly, 4),
                "kelly_bet_fraction": round(kelly_frac, 4),
                "expected_roi": round(roi, 2),
                "mc_std": round(float(mc["std_probs"][i]) * 100, 2),
                "prob_z": round(float(prob_z[i]), 3),
            }
            results.append(result)
        
        results.sort(key=lambda x: x["model_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        bases = results[:2]
        outsiders = [r for r in results[2:] if r["model_prob"] > 2.5]
        outsiders.sort(key=lambda x: x["value_ratio"], reverse=True)
        outsiders = outsiders[:3]
        
        # NEW: Analyze Trios
        trios = analyze_trios(results, mc, mc["all_probs"])
        
        # NEW: Find best place bet
        best_place = find_best_place_bet(results)
        
        sorted_p = sorted([r["model_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100.0, round(45.0 + gap * 2.2, 1))
        else:
            conf_idx = 50.0
        
        avg_vol = float(mc["vol_per_horse"].mean())
        vol_idx = min(100.0, round(avg_vol * 55.0, 1))
        
        if has_odds:
            raw_overround = sum(1.0 / h["odds"] for h in horses if h.get("odds", 0) > 1.01)
            overround_pct = round((raw_overround - 1.0) * 100, 1)
        else:
            overround_pct = None
        
        execution_time = time.time() - start_time
        
        return {
            "results": results,
            "bases": bases,
            "outsiders": outsiders,
            "trios": trios,  # NEW
            "best_place": best_place,  # NEW
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "weights": weights,
            "execution_time": round(execution_time, 2),
        }
    
    except Exception as e:
        logger.error(f"Engine error: {str(e)}")
        raise

# =============================================================================
# STREAMLIT UI
# =============================================================================

def apply_css() -> None:
    st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #07071a 0%, #0d1b2a 40%, #12192b 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a, #07071a); }
h1, h2, h3 { color: #e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(f"""
<div style="text-align:center; padding: 22px 0;">
    <h1 style="font-size:2.8em; background: linear-gradient(90deg,#00ff88,#00b4d8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}
    </h1>
    <p style="color:#6b7fa3;">Advanced Betting: Trio + Place Analysis</p>
</div>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
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


def main() -> None:
    st.set_page_config(
        page_title=f"🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}",
        page_icon="🏇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    init_session_state()
    apply_css()
    render_header()
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        mc_iter = st.slider("MC Itérations", 500, 5000, CONFIG.MC_ITERATIONS, 250)
        mw = st.slider("Poids Marché", 0.0, 0.60, CONFIG.MARKET_WEIGHT, 0.05)
        vt = st.slider("Seuil Value", 1.05, 1.60, CONFIG.VALUE_THRESHOLD, 0.05)
    
    tab1, tab2 = st.tabs(["📥 Données", "📊 Résultats"])
    
    with tab1:
        st.markdown("## 🏁 Course")
        c1, c2, c3 = st.columns(3)
        with c1:
            race_type = st.selectbox("Type", CONFIG.RACE_TYPES, key="race_type_select")
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100, key="distance_input")
        with c3:
            discipline = st.text_input("Prix", key="discipline_input")
        
        st.markdown("---\n## 🐎 Données Chevaux")
        
        input_method = st.radio("Méthode d'entrée:", ["Tableau Éditable", "Coller (Excel/Texte)"], horizontal=True, key="input_method")
        
        st.markdown("---")
        
        if input_method == "Tableau Éditable":
            st.markdown("### Tableau Éditable")
            st.info("💡 Modifiez directement | Les données sont sauvegardées automatiquement")
            
            edited_df = st.data_editor(
                st.session_state.horses_data,
                use_container_width=True,
                num_rows="dynamic",
                key="data_editor_main",
                height=400,
            )
            
            if edited_df is not None:
                st.session_state.horses_data = edited_df.copy()
            
            horses_input = st.session_state.horses_data.copy()
        
        else:
            st.markdown("### Coller depuis Excel/Texte")
            
            paste_data = st.text_area(
                "Collez vos données:",
                height=200,
                key="paste_area",
                placeholder="N°\tNom\tCote\tMusique Cheval\tMusique Driver\tMusique Entraîneur\tCorde"
            )
            
            if st.button("📥 Charger les données", key="load_paste_btn"):
                try:
                    lines = paste_data.strip().split("\n")
                    if len(lines) < 2:
                        st.error("❌ Données insuffisantes")
                        return
                    
                    data_rows = []
                    for line in lines[1:]:
                        parts = line.split("\t")
                        if len(parts) >= 7:
                            data_rows.append({
                                "N°": int(parts[0]),
                                "Nom": parts[1],
                                "Cote": float(parts[2]),
                                "Musique Cheval": parts[3],
                                "Musique Driver": parts[4],
                                "Musique Entraîneur": parts[5],
                                "Corde": int(parts[6]) if len(parts) > 6 else 0,
                            })
                    
                    if data_rows:
                        st.session_state.horses_data = pd.DataFrame(data_rows)
                        st.success(f"✅ {len(data_rows)} chevaux chargés!")
                        st.rerun()
                    else:
                        st.error("❌ Format invalide")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
            
            horses_input = st.session_state.horses_data.copy()
        
        st.markdown("---")
        
        if st.button("🚀 ANALYSER", use_container_width=True, key="analyze_btn"):
            if len(horses_input) < 2:
                st.error("❌ Minimum 2 partants")
                return
            
            horses_list = []
            for idx, row in horses_input.iterrows():
                try:
                    horses_list.append({
                        "number": int(row["N°"]),
                        "name": str(row["Nom"]),
                        "odds": float(row["Cote"]) if row["Cote"] > 0 else 0,
                        "horse_music": str(row["Musique Cheval"]),
                        "driver_music": str(row["Musique Driver"]),
                        "trainer_music": str(row["Musique Entraîneur"]),
                        "draw": int(row["Corde"]) if "Corde" in row else 0,
                    })
                except Exception as e:
                    st.error(f"❌ Ligne {idx}: {str(e)}")
                    return
            
            with st.spinner("Analyse en cours..."):
                try:
                    pred = run_engine_final(
                        {"race_type": race_type, "distance": int(distance), "discipline": discipline},
                        horses_list,
                        mc_iter=mc_iter, market_weight=mw, value_threshold=vt
                    )
                    st.session_state["prediction"] = pred
                    st.session_state["race_info"] = {"race_type": race_type, "distance": distance}
                    st.success(f"✅ Analyse réussie en {pred['execution_time']}s")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    with tab2:
        if "prediction" not in st.session_state:
            st.info("💡 Lancez l'analyse depuis l'onglet Données")
        else:
            pred = st.session_state["prediction"]
            
            # KPIs
            st.markdown("## 📊 KPIs")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Confiance", f"{pred['confidence_idx']}/100")
            with k2:
                st.metric("Volatilité", f"{pred['volatility_idx']}/100")
            with k3:
                st.metric("Partants", len(pred["results"]))
            with k4:
                vb = sum(1 for r in pred["results"] if r["is_value_bet"])
                st.metric("Value Bets", vb)
            
            st.markdown("---\n## 🏆 Classement")
            
            res_df = []
            for r in pred["results"]:
                res_df.append({
                    "Rg": r["rank"],
                    "N°": r["number"],
                    "Nom": r["name"],
                    "Gagnant%": f"{r['model_prob']:.1f}",
                    "Placé%": f"{r['place_prob']:.1f}",
                    "Kelly%": f"{r['kelly_bet_fraction']*100:.2f}",
                    "ROI%": f"{r['expected_roi']:.1f}",
                    "Value": "🟢" if r["is_value_bet"] else ("🔴" if r["value_ratio"] < 1.0 else "⚪"),
                })
            
            st.dataframe(pd.DataFrame(res_df), use_container_width=True, hide_index=True)
            
            # NEW: Placé Rentable
            st.markdown("---\n## 🎯 Meilleur Cheval en PLACÉ (Rentable)")
            if pred["best_place"]:
                bp = pred["best_place"]
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("N°", bp["number"])
                with col2:
                    st.metric("Nom", bp["name"][:15])
                with col3:
                    st.metric("Placé%", f"{bp['place_prob']:.1f}")
                with col4:
                    st.metric("Cote Est.", bp["estimated_place_odds"])
                with col5:
                    st.metric("ROI Placé%", bp["expected_roi_place"])
                
                st.markdown(f"""
**Kelly Sizing:**
- Full Kelly: {bp['kelly_criterion']:.2%} du bankroll
- Safe Kelly (25%): {bp['kelly_bet_fraction']:.2%} du bankroll
- Sur $1000: {1000 * bp['kelly_bet_fraction']:.0f}€
""")
            
            # NEW: 10 Meilleurs Trios
            st.markdown("---\n## 🎲 TOP 10 Combinaisons TRIO")
            
            trio_df = []
            for t in pred["trios"]:
                trio_df.append({
                    "Rg": t["rank"],
                    "Trio": f"{t['numbers'][0]}-{t['numbers'][1]}-{t['numbers'][2]}",
                    "Prob%": t["prob_pct"],
                    "Cote Est.": t["estimated_odds"],
                    "ROI%": t["expected_roi"],
                    "P1%": t["p1"],
                    "P2%": t["p2"],
                    "P3%": t["p3"],
                })
            
            st.dataframe(pd.DataFrame(trio_df), use_container_width=True, hide_index=True)
            
            st.markdown("""
**Comment parier au Trio:**
- Probabilité = chance d'avoir exactement cet ordre
- Cote Estimée = basée sur probabilité MC
- ROI = rentabilité attendue
- Meilleur ROI = meilleure sélection
""")


if __name__ == "__main__":
    main()
