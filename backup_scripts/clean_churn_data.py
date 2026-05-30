"""
Customer Churn Data Cleaning Script
------------------------------------
Reads the raw CSV, runs a dedicated cleaning function for each column,
and writes a cleaned CSV ready for model training.

Usage:
    python clean_churn_data.py

Input  : RAW_PATH  (set below)
Output : OUTPUT_PATH (set below)
"""

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_PATH    = "path/to/your/raw_data.csv"
OUTPUT_PATH = "path/to/your/cleaned_data.csv"


# ── Loader ─────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from: {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Column-level cleaning functions
# ══════════════════════════════════════════════════════════════════════════

def clean_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    customer_id — identifier column, not a feature.
    - Verify no duplicate IDs (would mean duplicate rows).
    - Drop the column; it must not be fed to any model.
    """
    dupes = df["customer_id"].duplicated().sum()
    if dupes:
        print(f"  [WARN] customer_id: {dupes} duplicate(s) found — review before dropping.")
    df = df.drop(columns=["customer_id"])
    return df


def clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    age — continuous numeric.
    - Null out the impossible value of 1.0 (migration artifact).
    - Null any age outside the valid range [18, 100].
    - Flag rows where age was originally 18.0 exactly (possible export floor).
    - Flag rows where age was missing/impossible (before imputation).
    - Impute remaining NaNs with the column median.
    """
    # Flag rows that had the suspicious clamped floor value of 18.0
    df["age_is_floor"] = (df["age"] == 18.0).astype(int)

    # Null impossible or out-of-range values
    df.loc[df["age"] < 18, "age"] = np.nan
    df.loc[df["age"] > 100, "age"] = np.nan

    # Flag rows that are missing (including the ones just nulled)
    df["age_was_missing"] = df["age"].isna().astype(int)

    # Impute with median
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)

    return df


def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    gender — categorical nominal.
    - Standardise all spelling/case variants to: Male, Female, Other.
    - Fill blanks with 'Unknown' (do not guess gender from other fields).
    - One-hot encode; drop the 'Unknown' column to avoid dummy trap.
    """
    mapping = {
        "male": "Male", "m": "Male", "MALE": "Male",
        "female": "Female", "f": "Female",
        "other": "Other",
    }
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: mapping.get(x, x))   # apply mapping
    )

    # Catch anything not mapped to a known value
    known = {"Male", "Female", "Other"}
    df.loc[~df["gender"].isin(known), "gender"] = "Unknown"

    # One-hot encode (drop 'Unknown' as reference category)
    dummies = pd.get_dummies(df["gender"], prefix="gender")
    dummies = dummies.drop(columns=["gender_Unknown"], errors="ignore")
    df = pd.concat([df.drop(columns=["gender"]), dummies], axis=1)

    return df


def clean_tenure_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    tenure_months — continuous numeric.
    - Flag rows capped at the maximum of 120.0 months (export ceiling).
    - Flag rows at the minimum of 1.0 months (possible new-customer default).
    - No imputation needed — no missing values in this field.
    """
    df["tenure_is_capped"]       = (df["tenure_months"] == 120.0).astype(int)
    df["tenure_is_new_customer"] = (df["tenure_months"] == 1.0).astype(int)
    return df


def clean_contract_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    contract_type — categorical ordinal (commitment length has a natural order).
    - Confirm the three expected values exist.
    - Ordinal-encode: Month-to-month=0, One year=1, Two year=2.
      (Do NOT one-hot encode — the ordering is meaningful for the model.)
    """
    order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_type"] = df["contract_type"].str.strip().map(order)

    unmapped = df["contract_type"].isna().sum()
    if unmapped:
        print(f"  [WARN] contract_type: {unmapped} row(s) could not be mapped — check raw values.")

    return df


def clean_monthly_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    monthly_charges — continuous numeric.
    - Flag rows sitting at the suspicious $15.00 floor (possible plan default).
    - Flag rows that were missing before imputation.
    - Impute the one missing row with the column median.
    """
    df["monthly_charges_at_floor"] = (df["monthly_charges"] == 15.0).astype(int)
    df["monthly_charges_was_missing"] = df["monthly_charges"].isna().astype(int)

    median_mc = df["monthly_charges"].median()
    df["monthly_charges"] = df["monthly_charges"].fillna(median_mc)

    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    total_charges — continuous numeric.
    - Flag rows where total_charges is implausibly low vs monthly × tenure
      (tolerance: actual total must be at least 30% of expected).
    - For flagged rows, replace total_charges with monthly_charges × tenure_months
      since the recorded value is clearly wrong.
    - Impute any remaining NaNs with the column median.
    - Flag rows that were missing before imputation.
    """
    df["total_charges_was_missing"] = df["total_charges"].isna().astype(int)

    # Compute expected total from billing columns
    expected = df["monthly_charges"] * df["tenure_months"]

    # Flag where actual is less than 30% of expected (and expected > 0)
    suspect = (expected > 0) & (df["total_charges"] < 0.3 * expected)
    df["total_charges_suspect"] = suspect.astype(int)

    # Replace suspect values with the computed estimate
    df.loc[suspect, "total_charges"] = expected[suspect].round(2)

    # Impute any remaining NaN
    median_tc = df["total_charges"].median()
    df["total_charges"] = df["total_charges"].fillna(median_tc)

    # Derived feature: average monthly spend across the customer's lifetime
    df["charges_per_month_ratio"] = (
        df["total_charges"] / df["tenure_months"].replace(0, np.nan)
    ).round(2)

    return df


def clean_internet_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    internet_service — categorical nominal.
    - Replace the string literal 'nan' with a real NaN first (critical step).
    - Standardise case/spelling variants to: DSL, Fiber optic, No internet.
    - Assign remaining NaN rows to 'Unknown' and flag them.
    - One-hot encode; drop 'Unknown' column as reference.
    """
    # 'nan' as a string will NOT be caught by pandas — replace it explicitly
    df["internet_service"] = df["internet_service"].replace("nan", np.nan)

    mapping = {
        "dsl": "DSL",
        "fiber optic": "Fiber optic",
        "fiber": "Fiber optic",
        "no": "No internet",
        "none": "No internet",
        "no internet": "No internet",
    }
    df["internet_service"] = (
        df["internet_service"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: mapping.get(x, x))
    )

    # Flag and fill rows that are still NaN or unmapped
    known = {"DSL", "Fiber optic", "No internet"}
    df["internet_service_unknown"] = (~df["internet_service"].isin(known)).astype(int)
    df.loc[~df["internet_service"].isin(known), "internet_service"] = "Unknown"

    # One-hot encode
    dummies = pd.get_dummies(df["internet_service"], prefix="internet")
    dummies = dummies.drop(columns=["internet_Unknown"], errors="ignore")
    df = pd.concat([df.drop(columns=["internet_service"]), dummies], axis=1)

    return df


def clean_phone_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    phone_service — binary categorical.
    - Standardise Yes/No/Y/N/yes/no variants to 'Yes' or 'No'.
    - Binary-encode as 1/0 (no need for one-hot on a two-value column).
    """
    mapping = {
        "yes": 1, "y": 1,
        "no": 0,  "n": 0,
    }
    df["phone_service"] = (
        df["phone_service"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    unmapped = df["phone_service"].isna().sum()
    if unmapped:
        print(f"  [WARN] phone_service: {unmapped} row(s) could not be mapped.")

    return df


def clean_avg_monthly_gb_used(df: pd.DataFrame) -> pd.DataFrame:
    """
    avg_monthly_gb_used — continuous numeric.
    - Null any negative value (physically impossible).
    - Flag rows with extremely high usage (> 80 GB/month).
    - Flag rows that were missing/impossible before imputation.
    - Impute NaNs with the column median.
    """
    # Null negative values
    df.loc[df["avg_monthly_gb_used"] < 0, "avg_monthly_gb_used"] = np.nan

    # Flag high-usage outlier rows (keep the value; just mark them)
    df["gb_high_outlier"] = (df["avg_monthly_gb_used"] > 80).astype(int)

    # Flag rows that were missing
    df["gb_was_missing"] = df["avg_monthly_gb_used"].isna().astype(int)

    # Impute with median
    median_gb = df["avg_monthly_gb_used"].median()
    df["avg_monthly_gb_used"] = df["avg_monthly_gb_used"].fillna(median_gb)

    return df


def clean_num_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    num_support_tickets — discrete count.
    - No missing values; convert float to int.
    - Add a binary flag for customers who raised any ticket.
    - Add a flag for high-ticket customers (>= 4 tickets).
    """
    df["num_support_tickets"] = df["num_support_tickets"].fillna(0).astype(int)
    df["has_any_tickets"]  = (df["num_support_tickets"] > 0).astype(int)
    df["is_high_ticket"]   = (df["num_support_tickets"] >= 4).astype(int)
    return df


def clean_avg_monthly_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    avg_monthly_minutes — continuous numeric.
    - Flag rows with suspiciously low minutes (< 20) despite having phone service.
    - Flag rows with high usage (> 500 minutes/month).
    - Flag rows missing before imputation.
    - Impute NaNs with the column median.
    """
    # Flag low-usage rows (plausible but worth monitoring)
    df["minutes_low_usage"] = (
        (df["avg_monthly_minutes"] < 20) & (df["phone_service"] == 1)
    ).astype(int)

    # Flag high-usage rows
    df["minutes_high_user"] = (df["avg_monthly_minutes"] > 500).astype(int)

    # Flag missing rows before imputing
    df["minutes_was_missing"] = df["avg_monthly_minutes"].isna().astype(int)

    median_min = df["avg_monthly_minutes"].median()
    df["avg_monthly_minutes"] = df["avg_monthly_minutes"].fillna(median_min)

    return df


def clean_satisfaction_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    satisfaction_score — continuous numeric, assumed scale 0–10.
    - Null any value > 10 (impossible on a 0–10 scale; e.g. the 99.0 entry).
    - Flag rows that were missing/impossible before imputation.
    - Impute NaNs with the column median.
    - Add a binary 'dissatisfied' flag for scores <= 4.
    """
    # Null out-of-scale values
    df.loc[df["satisfaction_score"] > 10, "satisfaction_score"] = np.nan

    # Flag missing rows
    df["satisfaction_was_missing"] = df["satisfaction_score"].isna().astype(int)

    # Impute with median
    median_sat = df["satisfaction_score"].median()
    df["satisfaction_score"] = df["satisfaction_score"].fillna(median_sat)

    # Binary flag: dissatisfied customer (score <= 4)
    df["is_dissatisfied"] = (df["satisfaction_score"] <= 4).astype(int)

    return df


def clean_payment_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    payment_method — categorical nominal.
    - Standardise abbreviations: 'BT' -> 'Bank transfer', 'CC' -> 'Credit card'.
    - Standardise case variants.
    - Fill missing rows with 'Unknown' and flag them.
    - One-hot encode; drop 'Unknown' column as reference.
    """
    mapping = {
        "bank transfer": "Bank transfer",
        "bt": "Bank transfer",
        "credit card": "Credit card",
        "cc": "Credit card",
        "electronic check": "Electronic check",
        "mailed check": "Mailed check",
    }
    df["payment_method"] = (
        df["payment_method"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: mapping.get(x, x))
    )

    # Flag and fill unmapped / missing rows
    known = {"Bank transfer", "Credit card", "Electronic check", "Mailed check"}
    df["payment_method_unknown"] = (~df["payment_method"].isin(known)).astype(int)
    df.loc[~df["payment_method"].isin(known), "payment_method"] = "Unknown"

    # One-hot encode
    dummies = pd.get_dummies(df["payment_method"], prefix="payment")
    dummies = dummies.drop(columns=["payment_Unknown"], errors="ignore")
    df = pd.concat([df.drop(columns=["payment_method"]), dummies], axis=1)

    return df


def clean_num_additional_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    num_additional_services — discrete count, range 0–5, no missing values.
    - Convert to int.
    - Add a binary flag for any add-on subscription.
    - Add a flag for fully-subscribed customers (5 services).
    """
    df["num_additional_services"] = df["num_additional_services"].fillna(0).astype(int)
    df["has_any_addons"] = (df["num_additional_services"] > 0).astype(int)
    df["max_addons"]     = (df["num_additional_services"] == 5).astype(int)
    return df


def clean_last_interaction_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    last_interaction_date — raw date, not usable directly by ML models.
    - Parse to datetime.
    - Compute days_since_last_interaction relative to the latest date in the dataset.
    - Flag customers inactive for more than 90 days.
    - Drop the original date column.
    """
    df["last_interaction_date"] = pd.to_datetime(df["last_interaction_date"], errors="coerce")

    reference_date = df["last_interaction_date"].max()
    df["days_since_last_interaction"] = (
        (reference_date - df["last_interaction_date"]).dt.days
    )

    # Impute any NaT rows (failed parse) with the median recency
    median_days = df["days_since_last_interaction"].median()
    df["days_since_last_interaction"] = (
        df["days_since_last_interaction"].fillna(median_days).astype(int)
    )

    # Flag customers with no interaction for more than 90 days
    df["long_since_interaction"] = (df["days_since_last_interaction"] > 90).astype(int)

    # Drop the raw date column — models cannot use datetime objects
    df = df.drop(columns=["last_interaction_date"])

    return df


def clean_churned(df: pd.DataFrame) -> pd.DataFrame:
    """
    churned — target variable (0 = retained, 1 = churned).
    - Verify only 0 and 1 values exist.
    - No transformation applied; target is returned as-is.
    """
    invalid = ~df["churned"].isin([0, 1])
    if invalid.sum():
        print(f"  [WARN] churned: {invalid.sum()} row(s) have unexpected values — inspect before training.")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Cross-field logical conflict checks
#  These run AFTER all individual column cleaning is complete so that
#  every value they reference is already in its cleaned/encoded form.
#  Each function adds one or more binary flag columns (1 = conflict found).
#  Flags are passed to the model as features — do not drop them.
# ══════════════════════════════════════════════════════════════════════════

def check_phone_minutes_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL — phone_service vs avg_monthly_minutes.
    A customer with no phone service (0) should have zero call minutes.
    Any positive minutes value when phone_service=0 is a direct contradiction
    between two features the model uses simultaneously.
    """
    df["conflict_phone_minutes"] = (
        (df["phone_service"] == 0) & (df["avg_monthly_minutes"] > 0)
    ).astype(int)
    return df


def check_internet_gb_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL — internet_service vs avg_monthly_gb_used.
    A customer with no internet service cannot have data usage.
    internet_No internet = 1 means the customer has no internet plan.
    Any GB usage above zero alongside that flag is a hard conflict.
    """
    # After OHE, 'No internet' becomes 'internet_No internet'
    no_internet_col = "internet_No internet"
    if no_internet_col in df.columns:
        df["conflict_internet_gb"] = (
            (df[no_internet_col] == 1) & (df["avg_monthly_gb_used"] > 0)
        ).astype(int)
    else:
        df["conflict_internet_gb"] = 0
    return df


def check_satisfaction_churn_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL — satisfaction_score vs churned.
    A highly satisfied customer (score >= 8) who is also marked as churned
    is statistically rare and likely indicates a wrong label or a wrong score.
    These rows can mislead the model on its strongest churn signal.
    """
    df["conflict_satisfaction_churn"] = (
        (df["satisfaction_score"] >= 8) & (df["churned"] == 1)
    ).astype(int)
    return df


def check_no_service_with_addons(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL — no services at all but num_additional_services > 0.
    A customer with no internet AND no phone cannot logically have any add-ons.
    This is a hard impossibility — the add-on count is fabricated for this row.
    """
    no_internet_col = "internet_No internet"
    if no_internet_col in df.columns:
        df["conflict_no_service_addons"] = (
            (df[no_internet_col] == 1)
            & (df["phone_service"] == 0)
            & (df["num_additional_services"] > 0)
        ).astype(int)
    else:
        df["conflict_no_service_addons"] = 0
    return df


def check_new_customer_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL (bug fix) — tenure = 1.0 rows excluded from the 30% billing check.
    New customers in their first month legitimately have partial billing.
    The main pipeline's suspect-flag logic would incorrectly penalise these rows.
    This function overrides that flag for tenure=1 customers so they are not
    treated as billing errors during model training.
    """
    # Clear the suspect flag for first-month customers set by clean_total_charges
    df.loc[df["tenure_months"] == 1.0, "total_charges_suspect"] = 0
    return df


def check_internet_addons_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    HIGH — internet_service = No internet but num_additional_services > 0.
    Most add-ons (streaming, security, backup) require an active internet plan.
    Having add-ons without internet is almost always a data error.
    """
    no_internet_col = "internet_No internet"
    if no_internet_col in df.columns:
        df["conflict_internet_addons"] = (
            (df[no_internet_col] == 1) & (df["num_additional_services"] > 0)
        ).astype(int)
    else:
        df["conflict_internet_addons"] = 0
    return df


def check_contract_tenure_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    HIGH — contract_type = Two year (encoded as 2) but tenure_months < 3.
    A customer on a two-year contract with under 3 months of tenure is suspicious,
    especially when stacked on top of the tenure_is_new_customer flag.
    Contract type is one of the strongest churn predictors — a wrong contract
    label on a short-tenure row can mislabel a high-risk customer as low-risk.
    """
    df["conflict_contract_tenure"] = (
        (df["contract_type"] == 2) & (df["tenure_months"] < 3)
    ).astype(int)
    return df


def check_satisfaction_tickets_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    HIGH — satisfaction_score >= 8 but num_support_tickets >= 4.
    A highly satisfied customer who also raised 4+ support tickets is
    contradictory. Either the satisfaction was recorded before the issues
    escalated or one of the values is wrong. Sends mixed signals to the model
    on two important features at the same time.
    """
    df["conflict_satisfaction_tickets"] = (
        (df["satisfaction_score"] >= 8) & (df["num_support_tickets"] >= 4)
    ).astype(int)
    return df


def check_inactive_retained_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """
    HIGH — days_since_last_interaction > 90 but churned = 0.
    A customer labelled as retained who has had no interaction in over 90 days
    is a strong candidate for being a silent churner with an incorrect label.
    This is a label quality issue — the model will learn the wrong target for
    these rows if they are not flagged.
    """
    df["conflict_inactive_retained"] = (
        (df["days_since_last_interaction"] > 90) & (df["churned"] == 0)
    ).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(raw_path: str, output_path: str) -> pd.DataFrame:
    """
    Run all cleaning steps in the correct order and save the result.
    Scaling (StandardScaler) is intentionally left out here — apply it
    inside your train/test split to avoid leaking test-set statistics.
    """
    df = load_data(raw_path)

    print("\nCleaning columns...")

    # 1. Drop identifier — must go first
    df = clean_customer_id(df)

    # 2. Fix impossible values before any imputation runs
    df = clean_age(df)
    df = clean_avg_monthly_gb_used(df)
    df = clean_satisfaction_score(df)

    # 3. Normalise categoricals before encoding
    df = clean_gender(df)
    df = clean_internet_service(df)        # also replaces 'nan' string
    df = clean_phone_service(df)
    df = clean_payment_method(df)
    df = clean_contract_type(df)

    # 4. Numeric fields — flag then impute
    df = clean_tenure_months(df)
    df = clean_monthly_charges(df)
    df = clean_total_charges(df)           # depends on monthly_charges being clean
    df = clean_avg_monthly_minutes(df)     # depends on phone_service being encoded
    df = clean_num_support_tickets(df)
    df = clean_num_additional_services(df)

    # 5. Date engineering — drop raw date, create numeric recency feature
    df = clean_last_interaction_date(df)

    # 6. Validate target column (no transformation)
    df = clean_churned(df)

    # 7. Cross-field conflict checks — run after all columns are cleaned.
    #    Each adds a binary flag column the model can train on.
    print("\nRunning cross-field conflict checks...")

    # Critical checks
    df = check_phone_minutes_conflict(df)
    df = check_internet_gb_conflict(df)
    df = check_satisfaction_churn_conflict(df)
    df = check_no_service_with_addons(df)
    df = check_new_customer_total_charges(df)  # corrects a bug from clean_total_charges

    # High priority checks
    df = check_internet_addons_conflict(df)
    df = check_contract_tenure_conflict(df)
    df = check_satisfaction_tickets_conflict(df)
    df = check_inactive_retained_conflict(df)

    # Summary: print how many rows were flagged per conflict
    conflict_cols = [c for c in df.columns if c.startswith("conflict_")]
    for col in conflict_cols:
        count = df[col].sum()
        if count:
            print(f"  [FLAG] {col}: {count} row(s) flagged")

    print(f"\nCleaning complete. Final shape: {df.shape}")
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")

    return df


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cleaned_df = run_pipeline(RAW_PATH, OUTPUT_PATH)
