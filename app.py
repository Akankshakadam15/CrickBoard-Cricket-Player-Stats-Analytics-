"""
CrickBoard (Streamlit) — Cricket Player Stats & Highlights

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import db
import ml_features as ml

db.init_db()

DATA_PATH = Path(__file__).parent / "data" / "players.json"

st.set_page_config(
    page_title="CrickBoard",
    page_icon="🏏",
    layout="wide",
)

# ---------- THEME (scoreboard-ish) ----------
st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; }
    .score-box {
        background:#F5F5F5; border:1px solid rgba(20,20,20,0.12);
        border-radius:10px; padding:14px; text-align:center;
    }
    .score-value { font-family:monospace; font-weight:700; font-size:1.6rem; color:#C97A1E; }
    .score-label { font-size:0.7rem; letter-spacing:1px; color:rgba(20,20,20,0.6); text-transform:uppercase; }
    .highlight-box {
        background:linear-gradient(135deg,#1B4332,#2B5E3A);
        border-radius:10px; padding:20px; margin-top:10px;
    }
    .milestone-item { padding:6px 0; border-bottom:1px solid rgba(20,20,20,0.12); color:#1a1a1a; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- DATA ----------
@st.cache_data
def load_players():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


players = load_players()
players_by_id = {p["id"]: p for p in players}
players_by_name = {p["name"]: p for p in players}

if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None


def current_favorites():
    if st.session_state.user_id:
        return db.get_favorites(st.session_state.user_id)
    return st.session_state.favorites


def toggle_favorite(player_id):
    if st.session_state.user_id:
        db.toggle_favorite(st.session_state.user_id, player_id)
        return
    if player_id in st.session_state.favorites:
        st.session_state.favorites.remove(player_id)
    else:
        st.session_state.favorites.append(player_id)


# ---------- HELPERS ----------
def render_scoreboard(p):
    is_bowler = p["role"] == "Bowler"
    if is_bowler:
        cells = [
            ("Matches", p["stats"]["matches"]),
            ("Wickets", p["stats"]["wickets"]),
            ("Average", p["stats"]["average"]),
            ("Economy", p["stats"]["economy"]),
            ("Best", p["stats"]["bestBowling"]),
            ("5W Hauls", p["stats"]["fiveWickets"]),
        ]
    else:
        cells = [
            ("Matches", p["stats"]["matches"]),
            ("Runs", f"{p['stats']['runs']:,}"),
            ("Average", p["stats"]["average"]),
            ("Strike Rate", p["stats"]["strikeRate"]),
            ("100s", p["stats"]["hundreds"]),
            ("50s", p["stats"]["fifties"]),
        ]
    cols = st.columns(len(cells))
    for col, (label, value) in zip(cols, cells):
        col.markdown(
            f'<div class="score-box"><div class="score-value">{value}</div>'
            f'<div class="score-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def render_career_chart(p):
    data = p["careerByYear"]
    if not data:
        st.caption("No year-by-year data on record for this player.")
        return
    key = "runs" if "runs" in data[0] else "wickets"
    df = pd.DataFrame(data).set_index("year")
    st.line_chart(df[key], color="#E8A33D")


def render_profile(p):
    col1, col2 = st.columns([1, 6])
    with col1:
        st.markdown(
            f'<div style="width:64px;height:64px;border-radius:50%;background:{p["color"]};'
            f'display:flex;align-items:center;justify-content:center;font-weight:700;'
            f'font-size:1.2rem;color:#101826;">{p["initials"]}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.subheader(p["name"])
        st.caption(f'{p["country"]} · {p["role"]} · {p["battingStyle"]}')

    is_fav = p["id"] in current_favorites()
    if st.button(("★ Remove from favorites" if is_fav else "☆ Add to favorites"), key=f"fav_{p['id']}"):
        toggle_favorite(p["id"])
        st.rerun()

    st.markdown("#### Scoreboard")
    render_scoreboard(p)

    st.markdown("#### Current form — recent seasons (IPL)")
    if p["recentForm"]:
        st.write("  ".join(f"`{r}`" for r in p["recentForm"]))
    else:
        st.caption("No recent innings on record — this profile reflects a completed career.")

    st.markdown("#### Career trend")
    render_career_chart(p)

    anomalies = ml.detect_breakout_dip_seasons(p)
    flagged = [a for a in anomalies if a["type"] != "normal"]
    if flagged:
        st.caption("Seasons flagged as statistically unusual vs. this player's own career average:")
        badge_cols = st.columns(len(flagged))
        for col, a in zip(badge_cols, flagged):
            icon = "🚀" if a["type"] == "breakout" else "📉"
            col.markdown(f"{icon} **{a['year']}** — {a['type']} (z={a['z_score']})")

    consistency = ml.consistency_score(p)
    if consistency:
        st.markdown("#### Consistency score")
        st.caption(
            "Based on how much this player's yearly output swings around their own average — "
            "100 = extremely consistent season to season, 0 = highly boom-or-bust."
        )
        st.progress(consistency["score"] / 100, text=f"{consistency['score']} / 100")

    st.markdown("#### Scouting report")
    st.caption("Auto-generated from this player's own stats and trend — template-based, not an LLM.")
    all_clusters = ml.cluster_players(players)
    cluster_label = all_clusters.get(p["id"], {}).get("label")
    st.info(ml.generate_scouting_report(p, cluster_label=cluster_label))

    st.markdown("#### Milestones")
    for m in p["milestones"]:
        st.markdown(f'<div class="milestone-item">● {m}</div>', unsafe_allow_html=True)

    st.markdown("#### Best innings")
    query = quote_plus(p["bestInnings"]["youtubeSearch"])
    st.markdown(
        f"""
        <div class="highlight-box">
            <div style="font-weight:700;font-size:1.1rem;">{p['bestInnings']['title']}</div>
            <div style="opacity:0.85;margin-top:6px;">{p['bestInnings']['description']}</div>
            <a href="https://www.youtube.com/results?search_query={query}" target="_blank"
               style="display:inline-block;margin-top:14px;background:#E8A33D;color:#101826;
               padding:8px 16px;border-radius:8px;text-decoration:none;font-weight:600;">
               ▶ Watch highlights on YouTube
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def player_card_grid(player_list, key_prefix=""):
    cols = st.columns(4)
    for i, p in enumerate(player_list):
        with cols[i % 4]:
            with st.container(border=True):
                st.markdown(
                    f'<div style="width:44px;height:44px;border-radius:50%;background:{p["color"]};'
                    f'display:flex;align-items:center;justify-content:center;font-weight:700;'
                    f'color:#101826;">{p["initials"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{p['name']}**")
                st.caption(f'{p["country"]} · {p["role"]}')
                if st.button("View profile", key=f"{key_prefix}_view_{p['id']}"):
                    st.session_state.selected_player = p["id"]
                    st.rerun()


# ---------- SIDEBAR NAV ----------
st.sidebar.title("🏏 CrickBoard")

# ---------- ACCOUNT (login/signup) ----------
with st.sidebar.expander("👤 Account", expanded=st.session_state.user_id is None):
    if st.session_state.user_id:
        st.write(f"Logged in as **{st.session_state.username}**")
        st.caption("Your favorites now sync to this account across sessions.")
        if st.button("Log out"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
    else:
        st.caption("Log in to save favorites permanently — or skip this and use guest mode below.")
        tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

        with tab_login:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Log in", key="login_btn"):
                uid = db.authenticate(login_user, login_pass)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = login_user
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        with tab_signup:
            signup_user = st.text_input("Choose a username", key="signup_user")
            signup_pass = st.text_input("Choose a password", type="password", key="signup_pass")
            if st.button("Create account", key="signup_btn"):
                ok, message = db.register_user(signup_user, signup_pass)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

page = st.sidebar.radio(
    "Navigate",
    ["Search", "Compare", "Favorites", "Predictions", "Player Clusters",
     "Similar Players", "Team Builder", "Ask CrickBoard", "About"],
)

if "selected_player" not in st.session_state:
    st.session_state.selected_player = None


# ---------- SEARCH PAGE ----------
if page == "Search":
    st.title("Find any player. See the full innings.")

    query = st.text_input("Search a player", placeholder="e.g. Kohli, Bumrah, Smith…")

    if query:
        matches = [p for p in players if query.lower() in p["name"].lower()]
    else:
        matches = players[:8]

    if not matches:
        st.warning(f'No player matched "{query}". Try another name.')
    else:
        player_card_grid(matches, key_prefix="search")

    if st.session_state.selected_player:
        st.divider()
        render_profile(players_by_id[st.session_state.selected_player])


# ---------- COMPARE PAGE ----------
elif page == "Compare":
    st.title("Compare two players.")

    names = [p["name"] for p in players]
    col1, col2 = st.columns(2)
    name_a = col1.selectbox("Player A", [""] + names)
    name_b = col2.selectbox("Player B", [""] + names)

    if name_a and name_b:
        a, b = players_by_name[name_a], players_by_name[name_b]

        if (a["role"] == "Bowler") != (b["role"] == "Bowler"):
            st.error(
                "Batting and bowling stats aren't directly comparable — "
                "pick two batters/all-rounders or two bowlers."
            )
        else:
            is_bowler = a["role"] == "Bowler"
            if is_bowler:
                metrics = [
                    ("Matches", "matches", None),
                    ("Wickets", "wickets", "high"),
                    ("Average", "average", "low"),
                    ("Economy", "economy", "low"),
                ]
            else:
                metrics = [
                    ("Matches", "matches", None),
                    ("Runs", "runs", "high"),
                    ("Average", "average", "high"),
                    ("Strike Rate", "strikeRate", "high"),
                    ("100s", "hundreds", "high"),
                    ("50s", "fifties", "high"),
                ]

            rows = []
            for label, key, better in metrics:
                va, vb = a["stats"][key], b["stats"][key]
                rows.append({"Metric": label, a["name"]: va, b["name"]: vb})

            df = pd.DataFrame(rows).set_index("Metric")
            st.table(df)

    else:
        st.info("Pick two players above to compare their stats side by side.")


# ---------- FAVORITES PAGE ----------
elif page == "Favorites":
    st.title("Your bookmarked players.")
    if st.session_state.user_id:
        st.caption(f"Synced to your account ({st.session_state.username}) — persists across sessions.")
    else:
        st.caption("Guest mode — these favorites will reset when you close the app. Log in (sidebar) to save them permanently.")
    fav_players = [players_by_id[fid] for fid in current_favorites() if fid in players_by_id]
    if not fav_players:
        st.info("No favorites yet — open a player's profile from Search and tap ☆ Add to favorites.")
    else:
        player_card_grid(fav_players, key_prefix="fav")
        if st.session_state.selected_player:
            st.divider()
            render_profile(players_by_id[st.session_state.selected_player])


# ---------- PREDICTIONS PAGE ----------
elif page == "Predictions":
    st.title("Player performance prediction")
    st.caption(
        "A simple linear regression (year → runs/wickets) fitted per player on their own "
        "career history. With only a handful of seasons per player this is a working "
        "prototype, not a reliable forecast — treat the trend line as directional, not exact."
    )

    name = st.selectbox("Choose a player", [""] + [p["name"] for p in players])
    if name:
        p = players_by_name[name]
        result = ml.predict_next_season(p)
        if not result:
            st.warning("Not enough season history for this player to fit a trend line.")
        else:
            if not result["reliable"]:
                st.warning(
                    f"Only {len(result['years'])} season(s) on record for {p['name']} — "
                    "this prediction is a rough extrapolation, not a confident forecast."
                )

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result["years"], y=result["actual"],
                                      mode="markers+lines", name="Actual"))
            fig.add_trace(go.Scatter(x=result["years"], y=result["fitted"],
                                      mode="lines", name="Fitted trend", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=[result["next_year"]], y=[result["predicted_value"]],
                                      mode="markers", name="Predicted next season",
                                      marker=dict(size=14, symbol="star", color="#C97A1E")))
            fig.update_layout(
                xaxis_title="Year", yaxis_title=result["metric"].capitalize(),
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric(
                f"Predicted {result['metric']} in {result['next_year']}",
                result["predicted_value"],
            )


# ---------- PLAYER CLUSTERS PAGE ----------
elif page == "Player Clusters":
    st.title("Player role discovery (clustering)")
    st.caption(
        "Groups players purely from their stats using KMeans — no manual role labels are "
        "given to the model. Cluster names below are auto-generated from each group's "
        "centroid (e.g. high strike rate + high average → 'Power Anchor')."
    )

    n_clusters = st.slider("Number of clusters per group (batters / bowlers)", 2, 6, 4)
    clusters = ml.cluster_players(players, n_clusters=n_clusters)

    rows = []
    for p in players:
        c = clusters.get(p["id"])
        if c:
            rows.append({"Player": p["name"], "Role": p["role"], "Cluster label": c["label"]})
    df = pd.DataFrame(rows)

    label_filter = st.multiselect("Filter by cluster label", sorted(df["Cluster label"].unique()))
    if label_filter:
        df = df[df["Cluster label"].isin(label_filter)]

    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------- SIMILAR PLAYERS PAGE ----------
elif page == "Similar Players":
    st.title("Players similar to...")
    st.caption(
        "Cosine similarity over standardized career stats — finds players with the closest "
        "overall statistical profile within the same broad role (batters/all-rounders vs "
        "bowlers, since their stats aren't on the same scale)."
    )

    name = st.selectbox("Choose a player", [""] + [p["name"] for p in players], key="similarity_pick")
    if name:
        target = players_by_name[name]
        similar = ml.find_similar_players(target, players, top_n=5)
        if not similar:
            st.info("No comparable players found.")
        else:
            rows = [{"Player": s["player"]["name"], "Role": s["player"]["role"],
                     "Similarity": s["similarity"]} for s in similar]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------- TEAM BUILDER PAGE ----------
elif page == "Team Builder":
    st.title("Team balance analyzer")
    st.caption("Pick up to 11 players and see whether your XI is batting-heavy, bowling-heavy, or balanced.")

    names = st.multiselect("Pick your XI", [p["name"] for p in players], max_selections=11)
    if names:
        team = [players_by_name[n] for n in names]
        batters = sum(1 for p in team if p["role"] == "Batsman")
        bowlers = sum(1 for p in team if p["role"] == "Bowler")
        allrounders = sum(1 for p in team if p["role"] == "All-rounder")

        col1, col2, col3 = st.columns(3)
        col1.metric("Batters", batters)
        col2.metric("Bowlers", bowlers)
        col3.metric("All-rounders", allrounders)

        if bowlers < 3:
            st.warning("Fewer than 3 specialist bowlers — this XI may be bowling-light.")
        if batters < 4:
            st.warning("Fewer than 4 specialist batters — this XI may be batting-light.")

        # Radar chart: 5 team-level metrics, each scaled 0-100 against the whole player pool
        def scaled(values, pool):
            if not pool or max(pool) == min(pool):
                return 50
            return round(100 * (sum(values) / len(values) - min(pool)) / (max(pool) - min(pool)))

        avg_pool = [p["stats"]["average"] for p in players if p["role"] != "Bowler"]
        sr_pool = [p["stats"]["strikeRate"] for p in players if p["role"] != "Bowler"]
        eco_pool = [-p["stats"]["economy"] for p in players if p["role"] == "Bowler"]
        wkt_pool = [p["stats"]["wickets"] for p in players if p["role"] == "Bowler"]
        matches_pool = [p["stats"]["matches"] for p in players]

        team_batters = [p for p in team if p["role"] != "Bowler"]
        team_bowlers = [p for p in team if p["role"] == "Bowler"]

        batting_depth = scaled([p["stats"]["average"] for p in team_batters], avg_pool) if team_batters else 0
        power_hitting = scaled([p["stats"]["strikeRate"] for p in team_batters], sr_pool) if team_batters else 0
        bowling_economy = scaled([-p["stats"]["economy"] for p in team_bowlers], eco_pool) if team_bowlers else 0
        bowling_strike = scaled([p["stats"]["wickets"] for p in team_bowlers], wkt_pool) if team_bowlers else 0
        experience = scaled([p["stats"]["matches"] for p in team], matches_pool)

        categories = ["Batting depth", "Power hitting", "Bowling economy", "Bowling threat", "Experience"]
        values = [batting_depth, power_hitting, bowling_economy, bowling_strike, experience]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                       fill="toself", name="Your XI"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Recommended additions")
        rec = ml.recommend_for_team_gap(team, players)
        if rec["weakest_area"] is None:
            st.success("This XI looks well balanced across batting depth, power hitting, and bowling — no clear gap detected.")
        else:
            st.write(f"Your XI's weakest area looks like: **{rec['weakest_area']}**. Players who'd help most:")
            rec_rows = []
            for rp in rec["recommendations"]:
                if rp["role"] == "Bowler":
                    rec_rows.append({"Player": rp["name"], "Role": rp["role"],
                                      "Wickets": rp["stats"]["wickets"], "Economy": rp["stats"]["economy"]})
                else:
                    rec_rows.append({"Player": rp["name"], "Role": rp["role"],
                                      "Average": rp["stats"]["average"], "Strike Rate": rp["stats"]["strikeRate"]})
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select players above to see the team balance radar chart.")


# ---------- ASK CRICKBOARD (NLP QUERY) PAGE ----------
elif page == "Ask CrickBoard":
    st.title("Ask CrickBoard")
    st.caption(
        "A small rule-based query tool — it matches a fixed set of question patterns "
        "(keyword/regex matching, not a general LLM) and answers from the dataset."
    )
    query = st.text_input(
        "Ask a question",
        placeholder='e.g. "most runs in 2024", "highest average", "best economy"',
    )
    if query:
        answer, df = ml.answer_query(query, players)
        st.write(answer)
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)


# ---------- ABOUT PAGE ----------
elif page == "About":
    st.title("How this scoreboard works")
    st.markdown(
        """
CrickBoard is a self-contained Streamlit app: every player, stat, and career figure lives in
`data/players.json`, generated from a real Kaggle IPL dataset via `convert_csv.py`.

**Search** matches on player name. **Current form** shows a player's recent IPL season
totals — swap in a live cricket API (CricAPI, Cricbuzz via RapidAPI) to make this update
automatically; see the code comments in `app.py` for where to plug it in.

**Best innings highlight** opens a YouTube search for that specific knock rather than embedding
a guessed video ID, so the link is always accurate.

**Accounts & favorites**: sign up or log in from the "👤 Account" panel in the sidebar. Logged-in
favorites are stored permanently in a local SQLite database (`db.py` / `crickboard.db`) and sync
across sessions. Without logging in, favorites still work but only last for the current session
(guest mode).

**Predictions, clustering, similarity, consistency score, and Ask CrickBoard** are all powered
by `ml_features.py` — see that file's docstring for details on each method (linear regression,
KMeans, cosine similarity, coefficient-of-variation scoring, and rule-based query matching).
        """
    )