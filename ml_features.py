"""
ml_features.py — Machine learning / analytics features for CrickBoard.

Covers:
    1. Performance prediction (per-player linear regression, year -> runs/wickets)
    2. Player clustering (KMeans role discovery)
    3. Player similarity engine (cosine similarity over stat vectors)
    4. Consistency / Form score (coefficient of variation of yearly output)
    5. Rule-based NLP query over the player dataset

All functions take the already-loaded `players` list (the same list used by app.py,
loaded from data/players.json) and return plain Python data structures so they're
easy to render with Streamlit.
"""

import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. PERFORMANCE PREDICTION
# ---------------------------------------------------------------------------

def predict_next_season(player, min_years=3):
    """
    Fits a simple linear regression (year -> runs or wickets) for one player
    and predicts the next season's value.

    Returns a dict with the historical (year, actual) pairs, the predicted
    next year and value, and a `reliable` flag (False when there isn't much
    data to fit on — callers should show a caveat in that case rather than
    presenting the number as a confident forecast).
    """
    data = player["careerByYear"]
    if len(data) < 2:
        return None

    key = "runs" if "runs" in data[0] else "wickets"
    years = np.array([d["year"] for d in data]).reshape(-1, 1)
    values = np.array([d[key] for d in data])

    model = LinearRegression()
    model.fit(years, values)

    next_year = int(years.max()) + 1
    predicted = float(model.predict([[next_year]])[0])
    predicted = max(0, round(predicted))  # runs/wickets can't be negative

    # "actual vs predicted" for the years we do have, so the chart can show
    # how well the line fits the real history
    fitted = model.predict(years)

    return {
        "metric": key,
        "years": [int(y) for y in years.flatten()],
        "actual": [int(v) for v in values],
        "fitted": [round(float(v), 1) for v in fitted],
        "next_year": next_year,
        "predicted_value": predicted,
        "reliable": len(data) >= min_years,
    }


# ---------------------------------------------------------------------------
# 2. PLAYER CLUSTERING (role discovery)
# ---------------------------------------------------------------------------

def cluster_players(players, n_clusters=4):
    """
    Splits batters/all-rounders and bowlers into separate KMeans runs (their
    stat scales aren't comparable), then auto-labels each cluster based on
    where its centroid sits relative to the others — e.g. high strike rate +
    high average -> "Power Anchor", high wickets + low economy -> "Strike
    Bowler". Returns a dict: player_id -> {"cluster": int, "label": str}.
    """
    batters = [p for p in players if p["role"] != "Bowler"]
    bowlers = [p for p in players if p["role"] == "Bowler"]

    result = {}
    result.update(_cluster_group(batters, ["average", "strikeRate", "hundreds", "fifties"],
                                  n_clusters, is_bowling=False))
    result.update(_cluster_group(bowlers, ["average", "economy", "wickets"],
                                  n_clusters, is_bowling=True))
    return result


def _cluster_group(group, feature_keys, n_clusters, is_bowling):
    if len(group) < n_clusters:
        n_clusters = max(1, len(group))
    if not group:
        return {}

    X = np.array([[p["stats"][k] for k in feature_keys] for p in group])
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    label_names = _name_clusters(km.cluster_centers_, feature_keys, is_bowling)

    out = {}
    for p, cluster_id in zip(group, labels):
        out[p["id"]] = {"cluster": int(cluster_id), "label": label_names[cluster_id]}
    return out


def _name_clusters(centers, feature_keys, is_bowling):
    """Heuristically names each cluster centroid based on its relative stats."""
    names = {}
    if is_bowling:
        avg_idx, eco_idx, wkt_idx = feature_keys.index("average"), feature_keys.index("economy"), feature_keys.index("wickets")
        eco_rank = np.argsort(centers[:, eco_idx])  # low economy first
        wkt_rank = np.argsort(-centers[:, wkt_idx])  # high wickets first
        for i in range(len(centers)):
            if i == eco_rank[0]:
                names[i] = "Economical / Containing Bowler"
            elif i == wkt_rank[0]:
                names[i] = "Strike Bowler (wicket-taker)"
            else:
                names[i] = "All-purpose Bowler"
    else:
        sr_idx, avg_idx = feature_keys.index("strikeRate"), feature_keys.index("average")
        sr_rank = np.argsort(-centers[:, sr_idx])
        avg_rank = np.argsort(-centers[:, avg_idx])
        for i in range(len(centers)):
            high_sr = i in sr_rank[: max(1, len(centers) // 3)]
            high_avg = i in avg_rank[: max(1, len(centers) // 3)]
            if high_sr and high_avg:
                names[i] = "Power Anchor (high SR + high avg)"
            elif high_sr:
                names[i] = "Power Hitter (explosive, high strike rate)"
            elif high_avg:
                names[i] = "Anchor (consistent, high average)"
            else:
                names[i] = "Developing / Squad Player"
    return names


# ---------------------------------------------------------------------------
# 3. SIMILARITY ENGINE
# ---------------------------------------------------------------------------

def find_similar_players(target_player, players, top_n=5):
    """
    Returns the top_n players most similar to target_player, using cosine
    similarity over a standardized stat vector. Only compares within the
    same broad group (batters/all-rounders vs bowlers) since their stats
    aren't on the same scale.
    """
    is_bowler = target_player["role"] == "Bowler"
    group = [p for p in players if (p["role"] == "Bowler") == is_bowler and p["id"] != target_player["id"]]
    if not group:
        return []

    feature_keys = ["average", "economy", "wickets"] if is_bowler else ["average", "strikeRate", "hundreds", "fifties"]

    all_players = [target_player] + group
    X = np.array([[p["stats"][k] for k in feature_keys] for p in all_players])
    X_scaled = StandardScaler().fit_transform(X)

    target_vec = X_scaled[0:1]
    others_vec = X_scaled[1:]
    sims = cosine_similarity(target_vec, others_vec)[0]

    ranked = sorted(zip(group, sims), key=lambda x: x[1], reverse=True)
    return [{"player": p, "similarity": round(float(s), 3)} for p, s in ranked[:top_n]]


# ---------------------------------------------------------------------------
# 4. CONSISTENCY / FORM SCORE
# ---------------------------------------------------------------------------

def consistency_score(player):
    """
    Computes a 0-100 consistency index from the coefficient of variation
    (std / mean) of a player's year-by-year output. Lower variation -> a
    higher score. This flags "consistent performer" vs "one big season".
    """
    data = player["careerByYear"]
    if len(data) < 2:
        return None

    key = "runs" if "runs" in data[0] else "wickets"
    values = np.array([d[key] for d in data], dtype=float)
    mean = values.mean()
    if mean == 0:
        return {"score": 0, "cv": None, "seasons": len(data)}

    cv = values.std() / mean  # coefficient of variation
    # Map CV to a 0-100 score: cv=0 -> 100 (perfectly consistent), cv>=1.5 -> 0
    score = max(0, round(100 * (1 - min(cv, 1.5) / 1.5)))
    return {"score": score, "cv": round(float(cv), 2), "seasons": len(data)}


def consistency_leaderboard(players, role_filter=None, min_seasons=3, top_n=10):
    rows = []
    for p in players:
        if role_filter and p["role"] != role_filter:
            continue
        c = consistency_score(p)
        if c and c["seasons"] >= min_seasons:
            rows.append({"name": p["name"], "role": p["role"], **c})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


# ---------------------------------------------------------------------------
# 5. RULE-BASED NLP QUERY
# ---------------------------------------------------------------------------

def answer_query(query, players):
    """
    Very small rule-based query engine — matches a handful of question
    patterns with regex/keywords and answers them from the player dataset.
    Not a general NLP system; it's a fixed set of supported question shapes.
    Returns (answer_text, dataframe_or_None).
    """
    q = query.lower().strip()

    # "most runs [in <year>]"
    m = re.search(r"most runs(?:\s+in\s+(\d{4}))?", q)
    if m:
        year = m.group(1)
        return _top_stat_by_year(players, "runs", int(year) if year else None, "runs")

    # "most wickets [in <year>]"
    m = re.search(r"most wickets(?:\s+in\s+(\d{4}))?", q)
    if m:
        year = m.group(1)
        return _top_stat_by_year(players, "wickets", int(year) if year else None, "wickets")

    # "highest average"
    if "highest average" in q or "best average" in q:
        rows = sorted(
            [{"name": p["name"], "average": p["stats"]["average"], "role": p["role"]} for p in players],
            key=lambda r: r["average"], reverse=True,
        )[:10]
        return "Top players by career average:", pd.DataFrame(rows)

    # "highest strike rate" / "fastest scorer"
    if "strike rate" in q or "fastest scorer" in q:
        batters = [p for p in players if p["role"] != "Bowler"]
        rows = sorted(
            [{"name": p["name"], "strikeRate": p["stats"]["strikeRate"]} for p in batters],
            key=lambda r: r["strikeRate"], reverse=True,
        )[:10]
        return "Top players by career strike rate:", pd.DataFrame(rows)

    # "most hundreds" / "most centuries"
    if "hundred" in q or "centur" in q:
        rows = sorted(
            [{"name": p["name"], "hundreds": p["stats"]["hundreds"]} for p in players if p["role"] != "Bowler"],
            key=lambda r: r["hundreds"], reverse=True,
        )[:10]
        return "Most career centuries:", pd.DataFrame(rows)

    # "best economy" / "most economical"
    if "econom" in q:
        bowlers = [p for p in players if p["role"] == "Bowler"]
        rows = sorted(
            [{"name": p["name"], "economy": p["stats"]["economy"]} for p in bowlers],
            key=lambda r: r["economy"],
        )[:10]
        return "Most economical bowlers (career):", pd.DataFrame(rows)

    return (
        "Sorry, I can only answer a fixed set of questions right now — try things like "
        "\"most runs in 2024\", \"most wickets\", \"highest average\", \"highest strike rate\", "
        "\"most hundreds\", or \"best economy\".",
        None,
    )


def _top_stat_by_year(players, stat_key, year, label):
    rows = []
    for p in players:
        for season in p["careerByYear"]:
            if stat_key not in season:
                continue
            if year and season["year"] != year:
                continue
            rows.append({"name": p["name"], "year": season["year"], label: season[stat_key]})

    if not rows:
        return f"No data found for that query.", None

    df = pd.DataFrame(rows).sort_values(label, ascending=False).head(10).reset_index(drop=True)
    year_text = f" in {year}" if year else " (single-season high, any year)"
    return f"Top {label}{year_text}:", df


# ---------------------------------------------------------------------------
# 6. BREAKOUT / DIP SEASON DETECTOR (anomaly detection)
# ---------------------------------------------------------------------------

def detect_breakout_dip_seasons(player, z_threshold=1.0):
    """
    Flags statistically unusual seasons in a player's career using a z-score
    of each season's output against that player's own career mean/std —
    e.g. a season more than `z_threshold` standard deviations above their
    own average is flagged as a "breakout" season; well below average is
    flagged as a "dip" season (which may reflect injury, loss of form, a
    reduced role, etc. — the data alone can't say why).

    Returns a list of dicts: {"year": int, "value": num, "z_score": float,
    "type": "breakout" | "dip" | "normal"}.
    """
    data = player["careerByYear"]
    if len(data) < 3:
        return []

    key = "runs" if "runs" in data[0] else "wickets"
    values = np.array([d[key] for d in data], dtype=float)
    mean, std = values.mean(), values.std()

    results = []
    for d, v in zip(data, values):
        if std == 0:
            z = 0.0
        else:
            z = (v - mean) / std
        if z >= z_threshold:
            season_type = "breakout"
        elif z <= -z_threshold:
            season_type = "dip"
        else:
            season_type = "normal"
        results.append({"year": d["year"], "value": v, "z_score": round(float(z), 2), "type": season_type})

    return results


# ---------------------------------------------------------------------------
# 7. AUTO-GENERATED SCOUTING REPORT (template-based, not an LLM)
# ---------------------------------------------------------------------------

def generate_scouting_report(player, cluster_label=None):
    """
    Builds a short natural-language summary of a player from their stats,
    trend direction, consistency, and any breakout/dip seasons — using
    fixed sentence templates filled in with real numbers, not a language
    model. Returns a plain string paragraph.
    """
    name = player["name"]
    role = player["role"]
    stats = player["stats"]
    data = player["careerByYear"]

    sentences = []

    # Opening line: role + cluster label if available
    if cluster_label:
        sentences.append(f"{name} is best described as a **{cluster_label}** ({role}).")
    else:
        sentences.append(f"{name} plays as a {role.lower()}.")

    # Trend direction from simple linear fit
    if len(data) >= 2:
        key = "runs" if "runs" in data[0] else "wickets"
        years = [d["year"] for d in data]
        values = [d[key] for d in data]
        slope = np.polyfit(years, values, 1)[0]
        if slope > 0.5:
            trend_word = "an upward"
        elif slope < -0.5:
            trend_word = "a declining"
        else:
            trend_word = "a fairly flat"
        sentences.append(
            f"Across {len(data)} recorded IPL seasons, their {key} output shows {trend_word} trend."
        )

    # Consistency
    consistency = consistency_score(player)
    if consistency:
        if consistency["score"] >= 70:
            sentences.append(f"They've been a highly consistent performer season to season (consistency score: {consistency['score']}/100).")
        elif consistency["score"] <= 35:
            sentences.append(f"Their output swings noticeably from one season to the next (consistency score: {consistency['score']}/100) — a boom-or-bust profile rather than a steady one.")
        else:
            sentences.append(f"Their season-to-season output is moderately consistent (consistency score: {consistency['score']}/100).")

    # Breakout / dip seasons
    anomalies = detect_breakout_dip_seasons(player)
    breakouts = [a["year"] for a in anomalies if a["type"] == "breakout"]
    dips = [a["year"] for a in anomalies if a["type"] == "dip"]
    if breakouts:
        sentences.append(f"{'Season' if len(breakouts)==1 else 'Seasons'} {', '.join(map(str, breakouts))} stand out as clear breakout year(s), well above their own career average.")
    if dips:
        sentences.append(f"{'Season' if len(dips)==1 else 'Seasons'} {', '.join(map(str, dips))} were notably below their career average — possibly form, role changes, or fitness, which the stats alone can't confirm.")

    # Headline stat
    if role == "Bowler":
        sentences.append(f"Career: {stats['matches']} matches, {stats['wickets']} wickets at an average of {stats['average']} and an economy of {stats['economy']}.")
    else:
        sentences.append(f"Career: {stats['matches']} matches, {stats['runs']:,} runs at an average of {stats['average']} and a strike rate of {stats['strikeRate']}.")

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# 8. TEAM GAP RECOMMENDER (extends the Team Builder radar chart)
# ---------------------------------------------------------------------------

def recommend_for_team_gap(team, players, top_n=3):
    """
    Looks at a chosen XI (list of player dicts) and, for whichever area is
    weakest (batting depth, power hitting, bowling economy, bowling
    strike-rate), suggests specific players NOT already in the team who
    would improve that area the most.

    Returns: {"weakest_area": str, "recommendations": [player, ...]}
    """
    team_ids = {p["id"] for p in team}
    pool = [p for p in players if p["id"] not in team_ids]

    team_batters = [p for p in team if p["role"] != "Bowler"]
    team_bowlers = [p for p in team if p["role"] == "Bowler"]

    # crude 0-100 scores per area, same idea as the radar chart in app.py
    def avg(vals):
        return sum(vals) / len(vals) if vals else 0

    batting_depth = avg([p["stats"]["average"] for p in team_batters])
    power_hitting = avg([p["stats"]["strikeRate"] for p in team_batters])
    bowling_economy = -avg([p["stats"]["economy"] for p in team_bowlers]) if team_bowlers else -999
    bowling_strike = avg([p["stats"]["wickets"] for p in team_bowlers])
    num_bowlers = len(team_bowlers)
    num_batters = len(team_batters)

    # Decide the weakest area using simple heuristics against the whole pool's median
    all_batters = [p for p in players if p["role"] != "Bowler"]
    all_bowlers = [p for p in players if p["role"] == "Bowler"]

    med_avg = np.median([p["stats"]["average"] for p in all_batters]) if all_batters else 0
    med_sr = np.median([p["stats"]["strikeRate"] for p in all_batters]) if all_batters else 0
    med_eco = np.median([p["stats"]["economy"] for p in all_bowlers]) if all_bowlers else 0
    med_wkt = np.median([p["stats"]["wickets"] for p in all_bowlers]) if all_bowlers else 0

    gaps = {}
    if num_bowlers < 3:
        gaps["too few specialist bowlers"] = 3 - num_bowlers
    if num_batters < 4:
        gaps["too few specialist batters"] = 4 - num_batters
    if team_batters and batting_depth < med_avg:
        gaps["batting depth (low career average)"] = med_avg - batting_depth
    if team_batters and power_hitting < med_sr:
        gaps["power hitting (low strike rate)"] = med_sr - power_hitting
    if team_bowlers and -bowling_economy > med_eco:
        gaps["bowling economy (too expensive)"] = -bowling_economy - med_eco
    if team_bowlers and bowling_strike < med_wkt:
        gaps["bowling strike power (too few wickets)"] = med_wkt - bowling_strike

    if not gaps:
        return {"weakest_area": None, "recommendations": []}

    weakest_area = max(gaps, key=gaps.get)

    # pick the recommendation pool + sort key based on the weakest area
    if "bowler" in weakest_area and "few" in weakest_area:
        candidates = sorted([p for p in pool if p["role"] == "Bowler"],
                             key=lambda p: p["stats"]["wickets"], reverse=True)
    elif "batter" in weakest_area and "few" in weakest_area:
        candidates = sorted([p for p in pool if p["role"] != "Bowler"],
                             key=lambda p: p["stats"]["average"], reverse=True)
    elif "batting depth" in weakest_area:
        candidates = sorted([p for p in pool if p["role"] != "Bowler"],
                             key=lambda p: p["stats"]["average"], reverse=True)
    elif "power hitting" in weakest_area:
        candidates = sorted([p for p in pool if p["role"] != "Bowler"],
                             key=lambda p: p["stats"]["strikeRate"], reverse=True)
    elif "economy" in weakest_area:
        candidates = sorted([p for p in pool if p["role"] == "Bowler"],
                             key=lambda p: p["stats"]["economy"])
    else:  # bowling strike power
        candidates = sorted([p for p in pool if p["role"] == "Bowler"],
                             key=lambda p: p["stats"]["wickets"], reverse=True)

    return {"weakest_area": weakest_area, "recommendations": candidates[:top_n]}