"""
convert_csv.py — Converts the Kaggle "IPL Player Lifetime Statistics" CSV
into the players.json shape used by all three CrickBoard versions
(static HTML, Flask, Streamlit).

Usage:
    python convert_csv.py cricket_data_2026.csv players.json
"""

import csv
import json
import re
import sys


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s


def initials(name):
    parts = [p for p in name.split() if p]
    letters = "".join(p[0] for p in parts)[:3].upper()
    return letters or "PL"


def to_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def to_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def parse_high_score(val):
    """'113*' -> (113, True)   '45' -> (45, False)"""
    if not val:
        return 0, False
    not_out = val.strip().endswith("*")
    digits = val.strip().rstrip("*")
    return to_int(digits), not_out


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_by_player(rows):
    players = {}
    for row in rows:
        if row["Year"] == "No stats":
            continue
        players.setdefault(row["Player_Name"], []).append(row)
    return players


def build_player(name, seasons):
    # sort by year ascending for the career chart
    seasons = sorted(seasons, key=lambda r: to_int(r["Year"]))

    total_matches_bat = sum(to_int(r["Matches_Batted"]) for r in seasons)
    total_runs = sum(to_int(r["Runs_Scored"]) for r in seasons)
    total_hundreds = sum(to_int(r["Centuries"]) for r in seasons)
    total_fifties = sum(to_int(r["Half_Centuries"]) for r in seasons)
    total_balls_faced = sum(to_int(r["Balls_Faced"]) for r in seasons)

    total_matches_bowl = sum(to_int(r["Matches_Bowled"]) for r in seasons)
    total_wickets = sum(to_int(r["Wickets_Taken"]) for r in seasons)
    total_runs_conceded = sum(to_int(r["Runs_Conceded"]) for r in seasons)
    total_balls_bowled = sum(to_int(r["Balls_Bowled"]) for r in seasons)
    total_5w = sum(to_int(r["Five_Wicket_Hauls"]) for r in seasons)

    # role classification heuristic (dataset has no explicit role column)
    bats_a_lot = total_runs >= 400
    bowls_a_lot = total_wickets >= 20
    if bowls_a_lot and not bats_a_lot:
        role = "Bowler"
    elif bowls_a_lot and bats_a_lot:
        role = "All-rounder"
    else:
        role = "Batsman"

    matches = max(total_matches_bat, total_matches_bowl)
    total_not_outs = sum(to_int(r["Not_Outs"]) for r in seasons)
    dismissals = max(1, total_matches_bat - total_not_outs)
    batting_avg = round(total_runs / dismissals, 2) if total_matches_bat else 0
    strike_rate = round((total_runs / total_balls_faced) * 100, 2) if total_balls_faced else 0

    bowling_avg = round(total_runs_conceded / total_wickets, 2) if total_wickets else 0
    economy = round((total_runs_conceded / total_balls_bowled) * 6, 2) if total_balls_bowled else 0

    # best bowling figure across seasons, e.g. "5/18"
    best_bowling = "0/0"
    best_wkts = -1
    for r in seasons:
        fig = r.get("Best_Bowling_Match", "0/0")
        if "/" in fig:
            try:
                w = int(fig.split("/")[0])
            except ValueError:
                w = 0
            if w > best_wkts:
                best_wkts = w
                best_bowling = fig

    # career-by-year series for the trend chart
    career_by_year = []
    for r in seasons:
        entry = {"year": to_int(r["Year"])}
        if role == "Bowler":
            entry["wickets"] = to_int(r["Wickets_Taken"])
        else:
            entry["runs"] = to_int(r["Runs_Scored"])
        career_by_year.append(entry)

    # "recent form" -> last up to 5 seasons' totals (runs or wickets)
    recent_seasons = seasons[-5:]
    if role == "Bowler":
        recent_form = [to_int(r["Wickets_Taken"]) for r in recent_seasons]
    else:
        recent_form = [to_int(r["Runs_Scored"]) for r in recent_seasons]

    # best individual innings -> season with the highest single-match score
    best_year, best_score, best_not_out = None, -1, False
    for r in seasons:
        score, not_out = parse_high_score(r.get("Highest_Score", "0"))
        if score > best_score:
            best_score, best_not_out, best_year = score, not_out, to_int(r["Year"])

    if role == "Bowler":
        best_title = f"Best bowling figures of {best_bowling} in IPL {seasons[-1]['Year'] if seasons else ''}"
        best_desc = (
            f"{name}'s standout bowling season, taking {total_wickets} wickets across "
            f"their IPL career at an average of {bowling_avg}."
        )
    else:
        star = "*" if best_not_out else ""
        best_title = f"{best_score}{star} in IPL {best_year}"
        best_desc = (
            f"{name}'s highest individual IPL score on record, part of a career total of "
            f"{total_runs:,} runs at an average of {batting_avg}."
        )

    milestones = []
    if total_hundreds:
        milestones.append(f"{total_hundreds} IPL century/centuries")
    if total_fifties:
        milestones.append(f"{total_fifties} IPL half-centuries")
    if total_5w:
        milestones.append(f"{total_5w} five-wicket haul(s) in IPL")
    if total_wickets:
        milestones.append(f"{total_wickets} career IPL wickets at an average of {bowling_avg}")
    if total_runs:
        milestones.append(f"{total_runs:,} career IPL runs at a strike rate of {strike_rate}")
    if not milestones:
        milestones.append("Limited-sample IPL career on record in this dataset.")

    color_by_role = {
        "Batsman": "#E8A33D",
        "Bowler": "#14213D",
        "All-rounder": "#C0392B",
    }

    return {
        "id": slugify(name),
        "name": name,
        "country": "Not specified in dataset",
        "role": role,
        "battingStyle": "Not specified in dataset",
        "initials": initials(name),
        "color": color_by_role.get(role, "#2B5E3A"),
        "stats": (
            {
                "matches": matches,
                "wickets": total_wickets,
                "average": bowling_avg,
                "economy": economy,
                "bestBowling": best_bowling,
                "fiveWickets": total_5w,
            }
            if role == "Bowler"
            else {
                "matches": matches,
                "runs": total_runs,
                "average": batting_avg,
                "strikeRate": strike_rate,
                "hundreds": total_hundreds,
                "fifties": total_fifties,
            }
        ),
        "recentForm": recent_form,
        "careerByYear": career_by_year,
        "milestones": milestones,
        "bestInnings": {
            "title": best_title,
            "description": best_desc,
            "youtubeSearch": f"{name} {best_title} IPL highlights",
        },
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_csv.py <input_csv> <output_json> [min_matches]")
        sys.exit(1)

    csv_path, out_path = sys.argv[1], sys.argv[2]
    min_matches = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    rows = load_rows(csv_path)
    grouped = group_by_player(rows)

    players = []
    for name, seasons in grouped.items():
        p = build_player(name, seasons)
        if p["stats"]["matches"] >= min_matches:
            players.append(p)

    players.sort(key=lambda p: p["stats"].get("runs", 0) + p["stats"].get("wickets", 0) * 20, reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)

    print(f"Converted {len(players)} players (min_matches={min_matches}) -> {out_path}")


if __name__ == "__main__":
    main()
