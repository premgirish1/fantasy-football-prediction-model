

import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import nfl_data_py as nfl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="NFL Start/Sit Predictor",
    page_icon="🏈",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .start-badge {
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .sit-badge {
        background: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .stButton>button {
        background: #1e3a5f;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        rf     = joblib.load("rf_model.pkl")
        ridge  = joblib.load("ridge_model.pkl")
        meta   = joblib.load("model_meta.pkl")
        return rf, ridge, meta
    except FileNotFoundError:
        return None, None, None

rf, ridge, meta = load_models()

def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[.\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data(ttl=3600, show_spinner="Fetching latest NFL data...")
def get_live_data(season: int, feature_cols: list):
    wk_live = (
        nfl.import_weekly_data([season], downcast=True)
            .sort_values(["player_id", "season", "week"])
            .query("season_type == 'REG' and position in ['RB','WR']")
            .copy()
    )
    g = wk_live.groupby("player_id", group_keys=False)

    # Rolling features
    for col in ["targets", "receptions", "receiving_yards", "rushing_yards", "carries", "target_share"]:
        if col in wk_live.columns:
            wk_live[f"roll3_{col}"]  = g[col].shift(1).rolling(3, min_periods=1).mean()
            wk_live[f"{col}_lag1"]   = g[col].shift(1)
            wk_live[f"delta_{col}"]  = wk_live[f"{col}_lag1"] - g[col].shift(2)

    def_allowed = (
        wk_live.groupby(["opponent_team", "season", "week", "position"], as_index=False)
               .agg(fp_allowed=("fantasy_points_ppr", "sum"))
               .rename(columns={"opponent_team": "defense"})
    )
    def_pivot = (
        def_allowed.pivot_table(
            index=["defense", "season", "week"],
            columns="position", values="fp_allowed", fill_value=0
        ).reset_index()
    )
    def_pivot.columns.name = None
    def_pivot = def_pivot.rename(columns={"RB": "def_fp_allowed_RB", "WR": "def_fp_allowed_WR"})
    gdef = def_pivot.sort_values(["defense", "season", "week"]).groupby("defense", group_keys=False)
    for col in ["def_fp_allowed_RB", "def_fp_allowed_WR"]:
        if col in def_pivot.columns:
            def_pivot[f"{col}_roll3"] = gdef[col].shift(1).rolling(3, min_periods=1).mean()

    def_merge_cols = [c for c in ["defense", "season", "week",
                                   "def_fp_allowed_RB_roll3",
                                   "def_fp_allowed_WR_roll3"] if c in def_pivot.columns]
    wk_live = wk_live.merge(
        def_pivot[def_merge_cols],
        left_on=["opponent_team", "season", "week"],
        right_on=["defense", "season", "week"],
        how="left"
    ).drop(columns="defense", errors="ignore")

    wk_live["missing_targets"] = 0.0
    wk_live["merge_name"] = wk_live["player_display_name"].map(norm)

    return wk_live


def predict_roster(player_names: list, wk_live: pd.DataFrame, model, feature_cols: list):
    """Given a list of player names, return projections + START/SIT decisions."""
    expected = list(model.feature_names_in_)
    CURRENT_WEEK = int(wk_live["week"].max())
    wk_current = wk_live[wk_live["week"] == CURRENT_WEEK].copy()

    for c in expected:
        if c not in wk_current.columns:
            wk_current[c] = 0.0

    X_live = wk_current[expected].fillna(0)
    wk_current["projected_ppr"] = model.predict(X_live)

    want = {norm(n) for n in player_names}
    roster_proj = wk_current[wk_current["merge_name"].isin(want)].copy()

    missing = want - set(roster_proj["merge_name"])
    if missing:
        wk_live_copy = wk_live.copy()
        last_rows = (
            wk_live_copy[wk_live_copy["merge_name"].isin(missing)]
            .sort_values(["merge_name", "week"])
            .groupby("merge_name", as_index=False)
            .tail(1)
            .copy()
        )
        if not last_rows.empty:
            for c in expected:
                if c not in last_rows.columns:
                    last_rows[c] = 0.0
            X_last = last_rows[expected].fillna(0)
            last_rows["projected_ppr"] = model.predict(X_last)
            last_rows["note"] = "Most recent appearance (Wk " + last_rows["week"].astype(int).astype(str) + ")"
            keep = [c for c in ["merge_name", "player_display_name", "position",
                                 "recent_team", "opponent_team", "projected_ppr", "note"]
                    if c in last_rows.columns]
            roster_proj = pd.concat([roster_proj, last_rows[keep]], ignore_index=True)

    if roster_proj.empty:
        return pd.DataFrame()

    roster_proj = roster_proj.sort_values("projected_ppr", ascending=False).reset_index(drop=True)
    threshold   = roster_proj["projected_ppr"].quantile(0.50)  # top 50% = START
    roster_proj["decision"] = roster_proj["projected_ppr"].apply(
        lambda x: "START" if x >= threshold else "SIT"
    )
    roster_proj["rank"] = range(1, len(roster_proj) + 1)
    return roster_proj



st.markdown('<div class="main-header">🏈 NFL Fantasy Start/Sit Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ML-powered weekly projections for RBs & WRs · Ridge & Random Forest models trained on 2018–2024 data</div>', unsafe_allow_html=True)

if rf is None:
    st.error("⚠️ Models not found. Run `python nfl_model.py` first to train and save the models.")
    st.stop()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/National_Football_League_logo.svg/200px-National_Football_League_logo.svg.png", width=80)
    st.markdown("## ⚙️ Settings")

    model_choice = st.selectbox("Model", ["Random Forest (recommended)", "Ridge Regression"])
    active_model = rf if "Random Forest" in model_choice else ridge

    season = st.selectbox("Season", [2024, 2023], index=0)

    st.markdown("---")
    st.markdown("### 📋 Your Roster")
    st.caption("Enter player names exactly (e.g. 'Saquon Barkley'). Up to 10 players.")

    default_players = [
        "Saquon Barkley", "Tyreek Hill", "CeeDee Lamb",
        "Christian McCaffrey", "DeVonta Smith"
    ]
    player_inputs = []
    for i in range(10):
        val = default_players[i] if i < len(default_players) else ""
        p = st.text_input(f"Player {i+1}", value=val, key=f"player_{i}", label_visibility="collapsed")
        if p.strip():
            player_inputs.append(p.strip())

    st.markdown("---")
    run_btn = st.button("🚀 Get Predictions", use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Random Forest MAE", f"{meta['rf_mae']:.2f} pts")
with col2:
    st.metric("Ridge MAE", f"{meta['ridge_mae']:.2f} pts")
with col3:
    st.metric("Start/Sit Accuracy", f"{meta['start_sit_acc']:.1%}")
with col4:
    st.metric("Training Seasons", "2018 – 2022")

st.divider()

if run_btn or True:
    if not player_inputs:
        st.info("Enter at least one player name in the sidebar to get predictions.")
    else:
        with st.spinner("Loading live NFL data and running predictions..."):
            try:
                wk_live = get_live_data(season, meta["feature_cols"])
                current_week = int(wk_live["week"].max())
                results = predict_roster(player_inputs, wk_live, active_model, meta["feature_cols"])
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

        if results.empty:
            st.warning("None of the entered player names were found in the dataset. Check spelling.")
        else:
            st.markdown(f"### 📊 Week {current_week} Projections — {season} Season")

            display_cols = {
                "rank": "#",
                "player_display_name": "Player",
                "position": "Pos",
                "recent_team": "Team",
                "opponent_team": "Opp",
                "projected_ppr": "Proj PPR",
                "decision": "Decision",
            }
            show = [c for c in display_cols.keys() if c in results.columns]
            table = results[show].rename(columns=display_cols).copy()
            table["Proj PPR"] = table["Proj PPR"].round(1)

            def style_decision(val):
                if val == "START":
                    return "background-color: #dcfce7; color: #166534; font-weight: bold;"
                elif val == "SIT":
                    return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
                return ""

            styled = table.style.applymap(style_decision, subset=["Decision"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            found_names = {norm(n) for n in results["player_display_name"].dropna().tolist()}
            not_found   = [n for n in player_inputs if norm(n) not in found_names]
            if not_found:
                st.warning(f"Could not find: **{', '.join(not_found)}** — check spelling or try full name.")

            st.divider()

            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown("#### Projected PPR Points by Player")
                chart_data = results[results["player_display_name"].notna()].copy()
                colors = ["#166534" if d == "START" else "#991b1b"
                          for d in chart_data["decision"]]
                fig, ax = plt.subplots(figsize=(8, max(3, len(chart_data) * 0.6)))
                bars = ax.barh(
                    chart_data["player_display_name"],
                    chart_data["projected_ppr"],
                    color=colors
                )
                ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=10)
                ax.set_xlabel("Projected PPR Points", fontsize=11)
                ax.set_title(f"Week {current_week} Projections", fontsize=13, fontweight="bold")
                ax.invert_yaxis()
                ax.spines[["top", "right"]].set_visible(False)
                # Legend
                from matplotlib.patches import Patch
                legend = [Patch(color="#166534", label="START"), Patch(color="#991b1b", label="SIT")]
                ax.legend(handles=legend, loc="lower right", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with right_col:
                st.markdown("#### Feature Importances")
                fi = pd.Series(rf.feature_importances_, index=rf.feature_names_in_).sort_values(ascending=True).tail(12)
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                fi.plot(kind="barh", ax=ax2, color="#2563eb")
                ax2.set_title("Top Features (RF)", fontsize=12, fontweight="bold")
                ax2.spines[["top", "right"]].set_visible(False)
                ax2.set_xlabel("Importance")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

st.divider()
st.caption("Built with scikit-learn + nfl_data_py · Model trained on NFL data 2018–2024 · Data refreshes hourly")