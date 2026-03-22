
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nfl_data_py as nfl
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

YEARS = list(range(2018, 2025))
print("Loading weekly data...")
wk = nfl.import_weekly_data(YEARS, downcast=True)
wk = wk.query("season_type == 'REG' and position in ['RB', 'WR']").copy()

if "fumbles_lost" not in wk.columns:
    wk["fumbles_lost"] = 0

def ppr(df):
    return (
        df["receptions"].fillna(0)
        + df["receiving_yards"].fillna(0) / 10.0
        + df["receiving_tds"].fillna(0) * 6.0       # FIX: was rushing_yards * 6
        + df["rushing_yards"].fillna(0) / 10.0
        + df["rushing_tds"].fillna(0) * 6.0
        + df["fumbles_lost"].fillna(0) * -2.0
    )

wk["ppr_label"] = ppr(wk)
wk["label_fp"]  = wk["fantasy_points_ppr"].fillna(0)   # use official PPR as target

wk = wk.sort_values(["player_id", "season", "week"]).copy()
grp = wk.groupby("player_id", group_keys=False)

STAT_COLS = ["targets", "receptions", "receiving_yards",
             "rushing_yards", "carries", "target_share"]

for col in STAT_COLS:
    if col in wk.columns:
        wk[f"{col}_lag1"]  = grp[col].shift(1)
        wk[f"roll3_{col}"] = grp[col].shift(1).rolling(3, min_periods=1).mean()
        wk[f"delta_{col}"] = wk[f"{col}_lag1"] - grp[col].shift(2)

def_allowed = (
    wk.groupby(["opponent_team", "season", "week", "position"], as_index=False)
      .agg(fp_allowed=("label_fp", "sum"))
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
    def_pivot[f"{col}_roll3"] = gdef[col].shift(1).rolling(3, min_periods=1).mean()

wk = wk.merge(
    def_pivot[["defense", "season", "week", "def_fp_allowed_RB_roll3", "def_fp_allowed_WR_roll3"]],
    left_on=["opponent_team", "season", "week"],
    right_on=["defense", "season", "week"],
    how="left"
).drop(columns="defense")

print("Loading injury data...")
inj = nfl.import_injuries(range(2018, 2025)).copy()

mask_out = pd.Series(False, index=inj.index)
if "report_status" in inj.columns:
    mask_out |= inj["report_status"].isin(["Out", "Doubtful"])
if "practice_status" in inj.columns:
    mask_out |= inj["practice_status"].isin(["Did Not Practice"])

inj_out = inj.loc[mask_out].copy()
keys = ["gsis_id", "season", "week"]
if "gsis_id" in inj_out.columns:
    inj_out = inj_out.dropna(subset=["gsis_id"])
inj_out = inj_out.drop_duplicates(subset=[k for k in keys if k in inj_out.columns])

usage = wk[["player_id", "season", "week", "recent_team", "roll3_targets"]].copy()

if "gsis_id" in inj_out.columns:
    miss = inj_out.merge(
        usage,
        left_on=["gsis_id", "season", "week"],
        right_on=["player_id", "season", "week"],
        how="left"
    )
else:
    miss = inj_out.merge(
        wk[["player_id", "player_display_name", "season", "week", "recent_team", "roll3_targets"]],
        left_on=["full_name", "season", "week", "team"],
        right_on=["player_display_name", "season", "week", "recent_team"],
        how="left"
    )

vacuum = (
    miss.groupby(["recent_team", "season", "week"], as_index=False)["roll3_targets"]
        .sum()
        .rename(columns={"roll3_targets": "missing_targets"})
)

wk = wk.merge(vacuum, on=["recent_team", "season", "week"], how="left")
wk["missing_targets"] = wk["missing_targets"].fillna(0.0)

FEATURE_COLS = [
    "roll3_targets", "roll3_receptions", "roll3_receiving_yards",
    "roll3_rushing_yards", "roll3_carries", "roll3_target_share",
    "targets_lag1", "receptions_lag1", "receiving_yards_lag1",
    "rushing_yards_lag1", "carries_lag1", "target_share_lag1",
    "delta_targets", "delta_receptions",
    "def_fp_allowed_RB_roll3", "def_fp_allowed_WR_roll3",
    "missing_targets",
]
feature_cols = [c for c in FEATURE_COLS if c in wk.columns]

X = wk[feature_cols].fillna(0)
y = wk["label_fp"].fillna(0)

train_mask = wk["season"] < 2023
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
ridge_mae   = mean_absolute_error(y_test, ridge_preds)
print(f"Ridge MAE: {ridge_mae:.2f}")

print("Training Random Forest (this takes ~30s)...")
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_mae   = mean_absolute_error(y_test, rf_preds)
print(f"Random Forest MAE: {rf_mae:.2f}")

compare = pd.DataFrame({"actual": y_test.values, "predicted": rf_preds})
threshold = compare["predicted"].quantile(0.70)
compare["decision"]      = compare["predicted"].apply(lambda x: "START" if x >= threshold else "SIT")
compare["actual_start"]  = compare["actual"].apply(
    lambda x: "START" if x >= compare["actual"].quantile(0.70) else "SIT"
)
acc = accuracy_score(compare["actual_start"], compare["decision"])
print(f"Start/Sit Accuracy: {acc:.2%}")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax, color="#2563eb")
ax.set_title("Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")

joblib.dump(rf,     "rf_model.pkl")
joblib.dump(ridge,  "ridge_model.pkl")
joblib.dump({
    "feature_cols": feature_cols,
    "rf_mae":       rf_mae,
    "ridge_mae":    ridge_mae,
    "start_sit_acc": acc,
    "threshold_quantile": 0.70,
}, "model_meta.pkl")
print("Models saved: rf_model.pkl | ridge_model.pkl | model_meta.pkl")
print("\nDone! Now run:  streamlit run dashboard.py")