# part1_analysis.py
# Assumes df_clean, lr_model, xgb_model, scaler, X_train, X_test, y_test
# are already available from the cleaning and training scripts.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from sklearn.metrics import confusion_matrix


# path to the saved model artifacts
MODEL_DIR = "/content/drive/MyDrive/AllianzData"

# reused across predict_churn calls so we don't reload from disk every time
_model_cache = {}


# ---------------------------------------------------------------------------

def build_quality_summary(raw_path, df_clean):
    "Prints a before/after table for every column that had a data quality issue."

    df_raw = pd.read_csv(raw_path)

    rows = [
        ("age",
         "impossible value 1.0, out-of-range ages, 1 missing",
         f"min={df_raw['age'].min():.1f}  nulls={df_raw['age'].isna().sum()}",
         f"min={df_clean['age'].min():.1f}  nulls={df_clean['age'].isna().sum()}"),

        ("gender",
         "9 spelling/case variants, 3 missing",
         f"{df_raw['gender'].nunique()} unique values  nulls={df_raw['gender'].isna().sum()}",
         f"{sum(c.startswith('gender_') for c in df_clean.columns)} OHE columns"),

        ("tenure_months",
         "capped at 120 (x2), floor at 1.0 (x5)",
         f"max={df_raw['tenure_months'].max():.0f}",
         "flags added: tenure_is_capped, tenure_is_new_customer"),

        ("contract_type",
         "ordinal strings, needed encoding",
         "Month-to-month / One year / Two year",
         "0 / 1 / 2"),

        ("monthly_charges",
         "$15 floor (x4), 1 missing",
         f"nulls={df_raw['monthly_charges'].isna().sum()}  floor rows={(df_raw['monthly_charges']==15).sum()}",
         f"nulls={df_clean['monthly_charges'].isna().sum()}  floor flagged"),

        ("total_charges",
         "implausibly low vs monthly x tenure on several rows",
         "some rows off by >70%",
         f"suspect rows corrected + flagged: {df_clean['total_charges_suspect'].sum() if 'total_charges_suspect' in df_clean.columns else 'N/A'}"),

        ("internet_service",
         "string 'nan' literal, 6+ spelling variants",
         str(df_raw["internet_service"].dropna().unique().tolist()),
         f"{sum(c.startswith('internet_') for c in df_clean.columns)} OHE columns"),

        ("phone_service",
         "6 variants: Yes/No/Y/N/yes/no",
         str(df_raw["phone_service"].unique().tolist()),
         "binary 0/1"),

        ("avg_monthly_gb_used",
         "impossible negative (-86.55), high outlier (84.63)",
         f"min={df_raw['avg_monthly_gb_used'].min():.2f}",
         f"min={df_clean['avg_monthly_gb_used'].min():.2f}  nulls=0"),

        ("avg_monthly_minutes",
         "4 missing, 9.8 min/month with phone_service=Yes",
         f"nulls={df_raw['avg_monthly_minutes'].isna().sum()}",
         f"nulls={df_clean['avg_monthly_minutes'].isna().sum()}"),

        ("satisfaction_score",
         "impossible 99.0, several values above 10",
         f"max={df_raw['satisfaction_score'].max():.1f}",
         f"max={df_clean['satisfaction_score'].max():.1f}  is_dissatisfied flag added"),

        ("payment_method",
         "BT/CC abbreviations, mixed case, 2 missing",
         f"{df_raw['payment_method'].nunique()} unique values  nulls={df_raw['payment_method'].isna().sum()}",
         f"{sum(c.startswith('payment_') for c in df_clean.columns)} OHE columns"),

        ("last_interaction_date",
         "raw dates not usable by ML",
         "datetime strings",
         "days_since_last_interaction (int)"),
    ]

    print(f"raw    : {len(df_raw)} rows  {len(df_raw.columns)} columns")
    print(f"cleaned: {len(df_clean)} rows  {len(df_clean.columns)} columns\n")

    col_w = 25
    iss_w = 50
    bef_w = 45

    header = f"{'column':<{col_w}} {'issue':<{iss_w}} {'before':<{bef_w}} after"
    print(header)
    print("-" * (col_w + iss_w + bef_w + 30))
    for col, issue, before, after in rows:
        print(f"{col:<{col_w}} {issue:<{iss_w}} {before:<{bef_w}} {after}")


# ---------------------------------------------------------------------------

def compute_feature_associations(df_clean):
    """Ranks features by point-biserial correlation with churn and prints the top 15."""
    # point-biserial is just Pearson when one variable is binary (0/1 target)
    # works for both continuous features and binary flag columns
    # p-value tells us if the correlation is statistically significant

    y = df_clean["churned"]
    results = []

    for col in df_clean.columns:
        if col == "churned" or df_clean[col].dtype == object:
            continue
        x = df_clean[col].fillna(df_clean[col].median())
        try:
            r, p = stats.pointbiserialr(x, y)
            results.append((col, round(r, 4), round(abs(r), 4), round(p, 4)))
        except Exception:
            continue

    results.sort(key=lambda x: x[2], reverse=True)

    print("point-biserial correlation with churn (top 15)\n")
    print(f"{'#':<4} {'feature':<35} {'r':>8}  {'|r|':>8}  {'p-value':>10}  sig?")
    print("-" * 72)
    for i, (col, r, absr, p) in enumerate(results[:15], 1):
        direction = "+" if r > 0 else "-"
        sig = "yes" if p < 0.05 else "no"
        print(f"{i:<4} {col:<35} {r:>8}  {absr:>8}  {p:>10}  {sig} ({direction})")

    print("\ntop 5:")
    for i, (col, r, absr, p) in enumerate(results[:5], 1):
        label = "higher churn" if r > 0 else "lower churn"
        print(f"  {i}. {col}  |r|={absr}  -> {label}")

    return pd.DataFrame(results, columns=["feature", "r", "abs_r", "p_value"])


# ---------------------------------------------------------------------------

def plot_eda_charts(df_clean):
    "Three bar charts showing churn rate by contract type, tenure, and satisfaction score."

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # chart 1 — churn by contract type
    ax = axes[0]
    label_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
    churn_ct = (
        df_clean.groupby("contract_type")["churned"]
        .mean().rename(index=label_map).mul(100)
    )
    churn_ct.plot(kind="bar", ax=ax, rot=15, color=["#d9534f", "#f0ad4e", "#5cb85c"])
    ax.set_ylabel("churn rate (%)")
    ax.set_title("Month-to-month customers churn 3x more than two-year contracts")
    for bar, val in zip(ax.patches, churn_ct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=9)

    # chart 2 — churn by tenure bucket
    ax = axes[1]
    df_clean = df_clean.copy()
    df_clean["tenure_bucket"] = pd.cut(
        df_clean["tenure_months"],
        bins=[0, 12, 24, 48, 72, 999],
        labels=["0-12m", "12-24m", "24-48m", "48-72m", "72m+"]
    )
    churn_ten = df_clean.groupby("tenure_bucket", observed=True)["churned"].mean().mul(100)
    churn_ten.plot(kind="bar", ax=ax, rot=15, color="#5b9bd5")
    ax.set_ylabel("churn rate (%)")
    ax.set_title("New customers (0-12m) are highest risk — early retention is critical")
    for bar, val in zip(ax.patches, churn_ten):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=9)

    # chart 3 — churn by satisfaction score
    ax = axes[2]
    df_clean["sat_bucket"] = pd.cut(
        df_clean["satisfaction_score"],
        bins=[0, 3, 5, 7, 10],
        labels=["0-3", "3-5", "5-7", "7-10"]
    )
    churn_sat = df_clean.groupby("sat_bucket", observed=True)["churned"].mean().mul(100)
    churn_sat.plot(kind="bar", ax=ax, rot=0,
                   color=["#d9534f", "#f0ad4e", "#f0ad4e", "#5cb85c"])
    ax.set_ylabel("churn rate (%)")
    ax.set_xlabel("satisfaction score band")
    ax.set_title("Satisfaction below 5 doubles churn — most actionable early warning signal")
    for bar, val in zip(ax.patches, churn_sat):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("eda_charts.png", dpi=130)
    plt.show()

    print("takeaways:")
    print("1. contract type is the clearest structural churn driver. month-to-month")
    print("   has no switching cost. moving customers to annual contracts is the")
    print("   single highest-impact retention offer.")
    print("2. churn is front-loaded in the first 12 months. onboarding quality")
    print("   directly determines whether a customer makes it past that window.")
    print("3. satisfaction score below 5 is a strong and actionable leading indicator.")
    print("   unlike tenure, it can be monitored in near real-time via surveys.")


# ---------------------------------------------------------------------------

def engineer_features(df_clean):
    """Adds two derived features and prints their stats and correlation with churn."""
    # feature 1: charges_per_month_ratio
    #   total_charges / tenure normalises lifetime spend by how long the customer
    #   has been active. a sudden spike vs historical average signals bill shock,
    #   a common churn trigger that raw total_charges alone won't show.
    #
    # feature 2: risk_score_composite
    #   weighted blend of satisfaction (inverted), contract type (inverted),
    #   and tenure (inverted). gives a single 0-1 risk score that retention reps
    #   can read directly, independent of the model probability.

    df = df_clean.copy()

    df["charges_per_month_ratio"] = (
        df["total_charges"] / df["tenure_months"].replace(0, np.nan)
    ).round(2).fillna(df["total_charges"].median())

    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0, index=s.index)

    df["risk_score_composite"] = (
        0.40 * (1 - minmax(df["satisfaction_score"])) +
        0.35 * (1 - minmax(df["contract_type"])) +
        0.25 * (1 - minmax(df["tenure_months"]))
    ).round(4)

    for feat in ["charges_per_month_ratio", "risk_score_composite"]:
        r, p = stats.pointbiserialr(df[feat].fillna(0), df["churned"])
        print(f"{feat}")
        print(f"  range: {df[feat].min():.3f} - {df[feat].max():.3f}")
        print(f"  nulls: {df[feat].isna().sum()}")
        print(f"  correlation with churn: r={r:.4f}  p={p:.4f}")
        print()

    return df


# ---------------------------------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test, model_name):
    "Plots a confusion matrix heatmap with counts and row percentages."

    cm  = confusion_matrix(y_test, model.predict(X_test))
    pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")

    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{model_name} — false negatives are missed churners (most costly)")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f"cm_{model_name.replace(' ', '_').lower()}.png", dpi=130)
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"true positives  (caught churners)     : {tp}")
    print(f"false negatives (missed churners)     : {fn}  <- costly")
    print(f"false positives (unnecessary outreach): {fp}")
    print(f"true negatives  (correctly retained)  : {tn}")


# ---------------------------------------------------------------------------

def export_model_artifacts(xgb_model, lr_model, scaler, feature_names, output_dir=MODEL_DIR):
    "Saves model, scaler, and feature list to disk as .pkl files."

    os.makedirs(output_dir, exist_ok=True)

    artifacts = {
        "churn_model_xgb.pkl": xgb_model,
        "churn_model_lr.pkl":  lr_model,
        "churn_scaler.pkl":    scaler,
        "churn_features.pkl":  feature_names,
    }

    for fname, obj in artifacts.items():
        path = f"{output_dir}/{fname}"
        joblib.dump(obj, path)
        print(f"saved {fname}  ({os.path.getsize(path) / 1024:.1f} KB)")


# ---------------------------------------------------------------------------

def predict_churn(customer_data: dict, model_dir=MODEL_DIR) -> dict:
    "Loads model artifacts once and returns churn probability, risk tier, and top 3 risk factors."

    if not _model_cache:
        _model_cache["model"]    = joblib.load(f"{model_dir}/churn_model_xgb.pkl")
        _model_cache["scaler"]   = joblib.load(f"{model_dir}/churn_scaler.pkl")
        _model_cache["features"] = joblib.load(f"{model_dir}/churn_features.pkl")

    model    = _model_cache["model"]
    scaler   = _model_cache["scaler"]
    features = _model_cache["features"]

    row  = pd.DataFrame([{f: customer_data.get(f, 0) for f in features}])
    prob = float(model.predict_proba(scaler.transform(row))[0][1])
    tier = "high" if prob >= 0.70 else "medium" if prob >= 0.40 else "low"

    top_idx     = np.argsort(model.feature_importances_)[::-1][:3]
    top_factors = [features[i] for i in top_idx]

    return {
        "churn_probability": round(prob, 4),
        "risk_tier":         tier,
        "top_risk_factors":  top_factors,
    }


def smoke_test_predict_churn(model_dir=MODEL_DIR):
    "Runs predict_churn on two test customers and validates the response schema."

    high_risk = {"contract_type": 0, "tenure_months": 2,
                 "satisfaction_score": 2.5, "monthly_charges": 95.0,
                 "num_support_tickets": 5}

    low_risk  = {"contract_type": 2, "tenure_months": 60,
                 "satisfaction_score": 8.5, "monthly_charges": 45.0,
                 "num_support_tickets": 0}

    for label, customer in [("high risk", high_risk), ("low risk", low_risk)]:
        r = predict_churn(customer, model_dir=model_dir)
        print(f"{label}: prob={r['churn_probability']}  tier={r['risk_tier']}")
        print(f"  top factors: {r['top_risk_factors']}")

    r = predict_churn(high_risk, model_dir=model_dir)
    assert isinstance(r["churn_probability"], float)
    assert r["risk_tier"] in ("high", "medium", "low")
    assert isinstance(r["top_risk_factors"], list) and len(r["top_risk_factors"]) == 3
    print("\nsmoke test passed — predict_churn() ready for part 2")
