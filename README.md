# CrickBoard — Cricket Player Stats & Analytics 

A Streamlit app for exploring real IPL player data: search, compare, career trends,
machine-learning-powered predictions, clustering, similarity search, a team balance
analyzer, a small natural-language query tool, and persistent accounts/favorites.

## Features

**Core**
- Search players by name
- Scoreboard-style stats (matches, runs/wickets, average, strike rate/economy, centuries)
- Current form — recent IPL season totals
- Career trend line chart
- Milestones list
- Best innings → accurate YouTube search link
- Compare two players side by side

**Machine learning / analytics**
- **Performance prediction** — per-player linear regression forecasting next season's runs/wickets
- **Player clustering** — KMeans auto-discovers role groups (Power Hitter, Anchor, Strike Bowler, etc.) purely from stats
- **Similarity engine** — "players similar to X", via cosine similarity over standardized stats
- **Consistency score** — 0-100 index from year-to-year variation, shown on each player's profile
- **Team Builder** — pick an XI, see a batting/bowling balance radar chart, and get specific player recommendations for your weakest area
- **Ask CrickBoard** — small rule-based query tool (e.g. "most runs in 2024", "highest average")
- **Breakout/Dip season detector** — flags statistically unusual seasons (z-score vs. a player's own career average) on each profile
- **Auto-generated scouting report** — a template-filled paragraph summary per player (role, trend, consistency, breakout/dip seasons, headline stats) — not an LLM, just real numbers in fixed sentence templates

**Accounts**
- Sign up / log in (SQLite-backed)
- Favorites sync permanently to your account when logged in, or work as session-only guest mode otherwise

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

It opens automatically in your browser, usually at `http://localhost:8501`.

A `crickboard.db` SQLite file is created automatically on first run, for accounts and favorites.
It's local to your machine — delete it any time to reset all accounts.

## Where the data comes from

`cricket_data_2026.csv` is the Kaggle "IPL Player Lifetime Statistics" dataset (season-by-season
stats per player). `convert_csv.py` aggregates it into the `players.json` shape the app uses:

```
python convert_csv.py cricket_data_2026.csv data/players.json [min_matches]
```

- `min_matches` (default 5) filters out players with very few recorded matches.
- **Role** (Batsman / Bowler / All-rounder), **best individual season score**, and **best bowling
  figures** are all derived automatically from the real season data.
- The dataset has no `country` or `battingStyle` fields, so those show as "Not specified in
  dataset" — fill them in by hand for any players you want to feature prominently.
- **"Current form"** shows a player's most recent IPL *season* totals (this dataset is
  season-level, not ball-by-ball) — the career trend chart shows the full multi-season history.

## Notes on the accounts system

Passwords are hashed with SHA-256 + a per-user random salt in `db.py` — fine for a student
project demo. A production app should use a proper password-hashing library (bcrypt or argon2)
instead, since SHA-256 is fast, which is actually a weakness for password hashing at scale.

## Notes on the ML features

- **Prediction** fits a linear regression per player on just their own season history (often
  only a handful of data points). Treat the output as a directional trend, not an accurate
  forecast — the app flags this explicitly when a player has fewer than 3 recorded seasons.
- **Clustering** and **similarity** compare batters/all-rounders separately from bowlers, since
  their stats aren't on the same scale, and standardize features before computing distances.
- **Ask CrickBoard** is a small fixed set of regex/keyword-matched question patterns, not a
  general NLP system — it only answers the question types listed on that page.
- **Breakout/dip detection** uses a z-score of each season vs. that player's own career mean —
  it flags *statistically* unusual seasons, but can't explain *why* (injury, role change, form)
  since that context isn't in the dataset.
- **Scouting reports** are built entirely from fixed sentence templates filled in with real
  computed numbers (trend slope, consistency score, breakout/dip years) — no language model is
  involved, so the wording is intentionally simple and repetitive across players.
- **Team gap recommender** compares your XI's batting/bowling stats against the whole player
  pool's median to guess the single weakest area, then ranks non-selected players by the stat
  that matters most for that gap. It's a heuristic, not a full optimization — it won't consider
  interactions between multiple simultaneous gaps.

## What sets this apart from typical cricket-analytics projects

Individual techniques here (regression-based prediction, KMeans clustering, a Streamlit
dashboard) are well established in cricket analytics research and open-source projects. What's
less commonly seen combined into one tool:
- All of search, prediction, clustering, similarity, consistency scoring, team building, and
  query answering in a single integrated app, rather than one technique in isolation
- The breakout/dip season detector and the auto-generated scouting report, which aren't common
  in the cricket-analytics projects/papers surveyed for this project
- The team recommender turning a static balance chart into an actionable "add this player" tool

