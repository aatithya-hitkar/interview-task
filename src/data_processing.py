import pandas as pd
import numpy as np

def load_data(path):
    "Reads a CSV file into a pandas DataFrame."
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def clean_customer_id(df):
    "Drops the customer identifier column since it is not a predictive feature."
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    return df

def clean_age(df):
    "Nullifies impossible ages, flags floor values, and imputes missing data with the median."
    df["age_is_floor"] = (df["age"] == 18.0).astype(int)
    df.loc[df["age"] < 18, "age"] = np.nan
    df.loc[df["age"] > 100, "age"] = np.nan
    df["age_was_missing"] = df["age"].isna().astype(int)
    df["age"] = df["age"].fillna(df["age"].median())
    return df

def clean_gender(df):
    "Standardizes gender labels and applies one-hot encoding."
    mapping = {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"}
    df["gender"] = df["gender"].astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, x))
    known = {"Male", "Female", "Other"}
    df.loc[~df["gender"].isin(known), "gender"] = "Unknown"
    dummies = pd.get_dummies(df["gender"], prefix="gender").drop(columns=["gender_Unknown"], errors="ignore")
    return pd.concat([df.drop(columns=["gender"]), dummies], axis=1)

def clean_tenure_months(df):
    "Flags rows with capped or minimum tenure values."
    df["tenure_is_capped"] = (df["tenure_months"] == 120.0).astype(int)
    df["tenure_is_new_customer"] = (df["tenure_months"] == 1.0).astype(int)
    return df

def clean_contract_type(df):
    "Ordinal-encodes the contract type based on commitment length."
    order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_type"] = df["contract_type"].str.strip().map(order)
    return df

def clean_monthly_charges(df):
    "Flags suspicious floor prices and imputes missing monthly charges."
    df["monthly_charges_at_floor"] = (df["monthly_charges"] == 15.0).astype(int)
    df["monthly_charges_was_missing"] = df["monthly_charges"].isna().astype(int)
    df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
    return df

def clean_total_charges(df):
    "Corrects implausibly low total charges and imputes missing values."
    df["total_charges_was_missing"] = df["total_charges"].isna().astype(int)
    expected = df["monthly_charges"] * df["tenure_months"]
    suspect = (expected > 0) & (df["total_charges"] < 0.3 * expected)
    df["total_charges_suspect"] = suspect.astype(int)
    df.loc[suspect, "total_charges"] = expected[suspect].round(2)
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
    return df

def clean_internet_service(df):
    "Standardizes internet service strings and applies one-hot encoding."
    df["internet_service"] = df["internet_service"].replace("nan", np.nan)
    mapping = {"dsl": "DSL", "fiber optic": "Fiber optic", "fiber": "Fiber optic", "no": "No internet", "none": "No internet", "no internet": "No internet"}
    df["internet_service"] = df["internet_service"].astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, x))
    known = {"DSL", "Fiber optic", "No internet"}
    df["internet_service_unknown"] = (~df["internet_service"].isin(known)).astype(int)
    df.loc[~df["internet_service"].isin(known), "internet_service"] = "Unknown"
    dummies = pd.get_dummies(df["internet_service"], prefix="internet").drop(columns=["internet_Unknown"], errors="ignore")
    return pd.concat([df.drop(columns=["internet_service"]), dummies], axis=1)

def clean_phone_service(df):
    "Binary-encodes the phone service column to 1 or 0."
    mapping = {"yes": 1, "y": 1, "no": 0, "n": 0}
    df["phone_service"] = df["phone_service"].astype(str).str.strip().str.lower().map(mapping)
    return df

def clean_avg_monthly_gb_used(df):
    "Nullifies negative usage values, flags high outliers, and imputes missing data."
    df.loc[df["avg_monthly_gb_used"] < 0, "avg_monthly_gb_used"] = np.nan
    df["gb_high_outlier"] = (df["avg_monthly_gb_used"] > 80).astype(int)
    df["gb_was_missing"] = df["avg_monthly_gb_used"].isna().astype(int)
    df["avg_monthly_gb_used"] = df["avg_monthly_gb_used"].fillna(df["avg_monthly_gb_used"].median())
    return df

def clean_num_support_tickets(df):
    "Fills missing ticket counts and adds flags for ticket presence and high volume."
    df["num_support_tickets"] = df["num_support_tickets"].fillna(0).astype(int)
    df["has_any_tickets"] = (df["num_support_tickets"] > 0).astype(int)
    df["is_high_ticket"] = (df["num_support_tickets"] >= 4).astype(int)
    return df

def clean_avg_monthly_minutes(df):
    "Flags anomalous phone minute usage and imputes missing values."
    df["minutes_low_usage"] = ((df["avg_monthly_minutes"] < 20) & (df["phone_service"] == 1)).astype(int)
    df["minutes_high_user"] = (df["avg_monthly_minutes"] > 500).astype(int)
    df["minutes_was_missing"] = df["avg_monthly_minutes"].isna().astype(int)
    df["avg_monthly_minutes"] = df["avg_monthly_minutes"].fillna(df["avg_monthly_minutes"].median())
    return df

def clean_satisfaction_score(df):
    "Removes out-of-scale scores, flags missing ones, imputes the median, and flags dissatisfaction."
    df.loc[df["satisfaction_score"] > 10, "satisfaction_score"] = np.nan
    df["satisfaction_was_missing"] = df["satisfaction_score"].isna().astype(int)
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())
    df["is_dissatisfied"] = (df["satisfaction_score"] <= 4).astype(int)
    return df

def clean_payment_method(df):
    "Standardizes payment methods and applies one-hot encoding."
    mapping = {"bank transfer": "Bank transfer", "bt": "Bank transfer", "credit card": "Credit card", "cc": "Credit card", "electronic check": "Electronic check", "mailed check": "Mailed check"}
    df["payment_method"] = df["payment_method"].astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, x))
    known = {"Bank transfer", "Credit card", "Electronic check", "Mailed check"}
    df["payment_method_unknown"] = (~df["payment_method"].isin(known)).astype(int)
    df.loc[~df["payment_method"].isin(known), "payment_method"] = "Unknown"
    dummies = pd.get_dummies(df["payment_method"], prefix="payment").drop(columns=["payment_Unknown"], errors="ignore")
    return pd.concat([df.drop(columns=["payment_method"]), dummies], axis=1)

def clean_num_additional_services(df):
    "Fills missing add-on counts and flags if customers have any or max services."
    df["num_additional_services"] = df["num_additional_services"].fillna(0).astype(int)
    df["has_any_addons"] = (df["num_additional_services"] > 0).astype(int)
    df["max_addons"] = (df["num_additional_services"] == 5).astype(int)
    return df

def clean_last_interaction_date(df):
    "Converts raw dates into days since last interaction and drops the original date column."
    df["last_interaction_date"] = pd.to_datetime(df["last_interaction_date"], errors="coerce")
    reference_date = df["last_interaction_date"].max()
    df["days_since_last_interaction"] = (reference_date - df["last_interaction_date"]).dt.days
    df["days_since_last_interaction"] = df["days_since_last_interaction"].fillna(df["days_since_last_interaction"].median()).astype(int)
    df["long_since_interaction"] = (df["days_since_last_interaction"] > 90).astype(int)
    return df.drop(columns=["last_interaction_date"])

def clean_churned(df):
    "Validates the churn target variable."
    invalid = ~df["churned"].isin([0, 1])
    if invalid.sum():
        print(f"Warning: {invalid.sum()} unexpected values in target column.")
    return df

def check_logical_conflicts(df):
    "Adds flags for cross-field logical contradictions found in the data."
    df["conflict_phone_minutes"] = ((df["phone_service"] == 0) & (df["avg_monthly_minutes"] > 0)).astype(int)
    if "internet_No internet" in df.columns:
        df["conflict_internet_gb"] = ((df["internet_No internet"] == 1) & (df["avg_monthly_gb_used"] > 0)).astype(int)
        df["conflict_no_service_addons"] = ((df["internet_No internet"] == 1) & (df["phone_service"] == 0) & (df["num_additional_services"] > 0)).astype(int)
        df["conflict_internet_addons"] = ((df["internet_No internet"] == 1) & (df["num_additional_services"] > 0)).astype(int)
    df["conflict_satisfaction_churn"] = ((df["satisfaction_score"] >= 8) & (df["churned"] == 1)).astype(int)
    df.loc[df["tenure_months"] == 1.0, "total_charges_suspect"] = 0
    df["conflict_contract_tenure"] = ((df["contract_type"] == 2) & (df["tenure_months"] < 3)).astype(int)
    df["conflict_satisfaction_tickets"] = ((df["satisfaction_score"] >= 8) & (df["num_support_tickets"] >= 4)).astype(int)
    df["conflict_inactive_retained"] = ((df["days_since_last_interaction"] > 90) & (df["churned"] == 0)).astype(int)
    return df

def run_cleaning_pipeline(raw_path):
    "Runs all cleaning steps in order and returns the fully processed DataFrame."
    df = load_data(raw_path)
    df = clean_customer_id(df)
    df = clean_age(df)
    df = clean_avg_monthly_gb_used(df)
    df = clean_satisfaction_score(df)
    df = clean_gender(df)
    df = clean_internet_service(df)
    df = clean_phone_service(df)
    df = clean_payment_method(df)
    df = clean_contract_type(df)
    df = clean_tenure_months(df)
    df = clean_monthly_charges(df)
    df = clean_total_charges(df)
    df = clean_avg_monthly_minutes(df)
    df = clean_num_support_tickets(df)
    df = clean_num_additional_services(df)
    df = clean_last_interaction_date(df)
    df = clean_churned(df)
    df = check_logical_conflicts(df)
    print(f"Cleaning complete. Final shape: {df.shape}")
    return df

def build_quality_summary(raw_path, df_clean):
    "Prints a before-and-after summary table for data quality issues."
    df_raw = pd.read_csv(raw_path)
    print(f"Raw shape: {df_raw.shape} | Cleaned shape: {df_clean.shape}\n")
    print(f"{'Column':<25} {'Issue':<50} {'Before':<45} {'After'}")
    print("-" * 130)
    print(f"{'age':<25} {'Impossible/missing values':<50} {'nulls=' + str(df_raw['age'].isna().sum()):<45} {'nulls=' + str(df_clean['age'].isna().sum())}")
    print(f"{'gender':<25} {'Spelling variants, missing data':<50} {str(df_raw['gender'].nunique()) + ' unique values':<45} {'OHE applied'}")
    print(f"{'total_charges':<25} {'Low vs expected billing':<50} {'Rows off by >70%':<45} {'Corrected & flagged'}")
    print(f"{'internet_service':<25} {'String nan, variants':<50} {str(df_raw['internet_service'].dropna().unique().tolist()):<45} {'OHE applied'}")
    print(f"{'satisfaction_score':<25} {'Values > 10':<50} {'max=' + str(df_raw['satisfaction_score'].max()):<45} {'max=' + str(df_clean['satisfaction_score'].max())}")

if __name__ == "__main__":
    raw_file = os.path.join(os.path.dirname(__file__), "..", "data", "test_datafile.csv")
    clean_file = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_datafile.csv")
    
    print("--- Starting Data Processing ---")
    df_cleaned = run_cleaning_pipeline(raw_file)
    build_quality_summary(raw_file, df_cleaned)
    
    df_cleaned.to_csv(clean_file, index=False)
    print(f"\n--- Saved cleaned data to {clean_file} ---")
