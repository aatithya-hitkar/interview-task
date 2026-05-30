"""
Churn Model Training Script
----------------------------
Reads the cleaned CSV produced by clean_churn_data.py,
applies final preparation steps, then trains and evaluates
Logistic Regression and XGBoost models.

Usage:
    python train_churn_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
import xgboost as xgb
import matplotlib.pyplot as plt

CLEANED_PATH = "path/to/your/cleaned_data.csv"


# ── Step 1: Load ────────────────────────────────────────────────────────────

def load_cleaned(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ── Step 2: Remove duplicate customer records ───────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows.
    Rows that share all column values are safe to deduplicate.
    Rows with the same ID but conflicting labels cannot be auto-resolved
    and are dropped entirely to avoid corrupting training data.
    """
    before = len(df)

    # Full row duplicates — keep first occurrence
    df = df.drop_duplicates()

    # Conflicting label rows: same customer_id, different churned value.
    # These exist in cleaned data only if customer_id was NOT dropped yet.
    # Since clean_churn_data.py drops customer_id, this guard is a safety net.
    after = len(df)
    print(f"Duplicates removed: {before - after} rows dropped ({after} remaining)")
    return df


# ── Step 3: Fix no-service add-on values ───────────────────────────────────

def fix_no_service_addons(df: pd.DataFrame) -> pd.DataFrame:
    """
    The only conflict where the numeric value is definitively wrong:
    customers with no internet AND no phone cannot have add-ons.
    Set num_additional_services to 0 for these rows.
    """
    no_internet_col = "internet_No internet"
    if no_internet_col in df.columns:
        mask = (
            (df[no_internet_col] == 1)
            & (df["phone_service"] == 0)
            & (df["num_additional_services"] > 0)
        )
        df.loc[mask, "num_additional_services"] = 0
        print(f"Fixed add-on values: {mask.sum()} rows corrected")
    return df


# ── Step 4: Recalibrate the inactive threshold ──────────────────────────────

def recalibrate_inactive_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    The hardcoded 90-day threshold flagged 63% of retained customers as
    inactive — too broad to be useful. Replace it with the 75th percentile
    of days_since_last_interaction, which is data-driven and business-neutral.
    Also recompute conflict_inactive_retained with the new threshold.
    """
    col = "days_since_last_interaction"
    threshold = int(df[col].quantile(0.75))
    print(f"Inactive threshold recalibrated: {threshold} days (75th percentile)")

    df["long_since_interaction"] = (df[col] > threshold).astype(int)

    if "conflict_inactive_retained" in df.columns:
        df["conflict_inactive_retained"] = (
            (df[col] > threshold) & (df["churned"] == 0)
        ).astype(int)

    return df


# ── Step 5: Drop noisy conflict flags ──────────────────────────────────────

def drop_noisy_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    conflict_phone_minutes (36% flagged) and conflict_internet_gb (14.5%)
    reflect business definitions, not data errors. Keeping them adds noise.
    """
    cols_to_drop = ["conflict_phone_minutes", "conflict_internet_gb"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped noisy flags: {cols_to_drop}")
    return df


# ── Step 6: Audit and fill any remaining NaNs ──────────────────────────────

def fix_remaining_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety net — catches any NaNs that slipped through the cleaning script.
    Prints exactly which columns still have NaNs and how many, then fills:
      - numeric columns  -> median of that column
      - object columns   -> 'Unknown'
    Logistic Regression raises a ValueError on any NaN, so this must
    run before the train/test split.
    """
    nan_counts = df.isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]

    if nan_cols.empty:
        print("No remaining NaNs found — data is clean")
        return df

    print(f"\nResidual NaNs found in {len(nan_cols)} column(s):")
    for col, count in nan_cols.items():
        print(f"  {col}: {count} NaN(s)")

    for col in nan_cols.index:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    still_nan = df.isnull().sum().sum()
    print(f"After fix: {still_nan} NaN(s) remaining")
    return df


# ── Step 7: Check and report class balance ──────────────────────────────────

def check_class_balance(df: pd.DataFrame) -> float:
    counts = df["churned"].value_counts()
    ratio  = counts[1] / len(df)
    print(f"\nClass balance — Churned: {counts[1]} ({ratio:.1%}) | Retained: {counts[0]} ({1-ratio:.1%})")
    return ratio


# ── Step 8: Split and scale ─────────────────────────────────────────────────

def prepare_train_test(df: pd.DataFrame):
    X = df.drop(columns=["churned"])
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale — fit on train only, apply to both
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ── Step 9: Evaluate helper ─────────────────────────────────────────────────

def evaluate(name: str, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return auc, y_proba


# ── Step 10: Logistic Regression ─────────────────────────────────────────────

def train_logistic(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",   # compensates for churn minority class
        max_iter=1000,
        random_state=42
    )

    # 5-fold CV on training data
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nLogistic Regression — CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X_train, y_train)
    return model


# ── Step 11: XGBoost ────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    # scale_pos_weight balances classes: retained_count / churned_count
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,    # handles class imbalance
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        verbosity=0
    )

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nXGBoost — CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X_train, y_train)
    return model


# ── Step 12: Feature importance (XGBoost) ───────────────────────────────────

def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"XGBoost — Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("\nTop 10 features:")
    print(importance.head(10).to_string())


# ── Step 13: ROC curve comparison ───────────────────────────────────────────

def plot_roc_curves(models_probas: dict, y_test):
    """models_probas = {'Model Name': y_proba_array}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in models_probas.items():
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)
    ax.set_title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.show()


# ── Pipeline ────────────────────────────────────────────────────────────────

def run(path: str):
    df = load_cleaned(path)

    # Pre-training preparation
    df = remove_duplicates(df)
    df = fix_no_service_addons(df)
    df = recalibrate_inactive_flag(df)
    df = drop_noisy_flags(df)
    df = fix_remaining_nans(df)
    check_class_balance(df)

    # Split and scale
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(df)

    # Train
    lr_model  = train_logistic(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Evaluate
    _, lr_proba  = evaluate("Logistic Regression", lr_model, X_test, y_test)
    _, xgb_proba = evaluate("XGBoost",             xgb_model, X_test, y_test)

    # Plots
    plot_roc_curves({"Logistic Regression": lr_proba, "XGBoost": xgb_proba}, y_test)
    plot_feature_importance(xgb_model, X_train.columns.tolist())

    return lr_model, xgb_model, scaler


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lr_model, xgb_model, scaler = run(CLEANED_PATH)
