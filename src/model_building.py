import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy import stats
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_model_cache = {}

def compute_feature_associations(df_clean):
    "Ranks features by their point-biserial correlation with the churn target and prints the top 15."
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
    print("Point-biserial correlation with churn (Top 15):")
    for i, (col, r, absr, p) in enumerate(results[:15], 1):
        print(f"{i}. {col}: r={r}, p={p}")
    return pd.DataFrame(results, columns=["feature", "r", "abs_r", "p_value"])

def plot_eda_charts(df_clean):
    "Generates and saves bar charts showing churn rate by contract type, tenure, and satisfaction."
    df_plot = df_clean.copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    churn_ct = df_plot.groupby("contract_type")["churned"].mean().mul(100)
    churn_ct.plot(kind="bar", ax=axes[0], title="Churn by Contract Type", color=["#d9534f", "#f0ad4e", "#5cb85c"])
    axes[0].set_ylabel("Churn Rate (%)")
    
    df_plot["tenure_bucket"] = pd.cut(df_plot["tenure_months"], bins=[0, 12, 24, 48, 72, 999], labels=["0-12m", "12-24m", "24-48m", "48-72m", "72m+"])
    churn_ten = df_plot.groupby("tenure_bucket", observed=True)["churned"].mean().mul(100)
    churn_ten.plot(kind="bar", ax=axes[1], title="Churn by Tenure", color="#5b9bd5")
    
    df_plot["sat_bucket"] = pd.cut(df_plot["satisfaction_score"], bins=[0, 3, 5, 7, 10], labels=["0-3", "3-5", "5-7", "7-10"])
    churn_sat = df_plot.groupby("sat_bucket", observed=True)["churned"].mean().mul(100)
    churn_sat.plot(kind="bar", ax=axes[2], title="Churn by Satisfaction Score", color=["#d9534f", "#f0ad4e", "#f0ad4e", "#5cb85c"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "..", "evaluation", "eda_charts.png"))
    plt.show()
    print("Saved eda_charts.png")

def engineer_features(df_clean):
    "Adds derived features for spending ratio and risk composite score."
    df = df_clean.copy()
    df["charges_per_month_ratio"] = (df["total_charges"] / df["tenure_months"].replace(0, np.nan)).round(2).fillna(df["total_charges"].median())
    
    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0, index=s.index)
        
    df["risk_score_composite"] = (0.40 * (1 - minmax(df["satisfaction_score"])) + 
                                  0.35 * (1 - minmax(df["contract_type"])) + 
                                  0.25 * (1 - minmax(df["tenure_months"]))).round(4)
    return df

def fix_remaining_nans(df):
    "Fills any remaining NaNs to prevent model training errors."
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")
    return df

def prepare_train_test(df):
    "Splits data into train and test sets and applies standard scaling."
    df = fix_remaining_nans(df.copy())
    
    # Remove target leakage columns before training
    leaky_cols = ["conflict_satisfaction_churn", "conflict_inactive_retained"]
    df = df.drop(columns=[c for c in leaky_cols if c in df.columns])
    
    X = df.drop(columns=["churned"])
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic(X_train, y_train):
    "Trains a class-balanced Logistic Regression model."
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    "Trains an XGBoost model explicitly on CPU adjusting for class imbalance."
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=ratio, eval_metric="auc", random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(name, model, X_test, y_test):
    "Evaluates the model using AUC and classification reports."
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{name} - AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    return y_proba

def plot_confusion_matrix(model, X_test, y_test, model_name):
    "Plots and saves a confusion matrix for the given model."
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained", "Churned"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Retained", "Churned"])
    ax.set_title(f"Confusion Matrix: {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i,j] < cm.max()/2 else "white")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "..", "evaluation", f"cm_{model_name.replace(' ', '_').lower()}.png"))
    plt.show()

def plot_roc_curves(models_probas, y_test):
    "Plots and saves ROC curves for multiple models."
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in models_probas.items():
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)
    ax.set_title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "..", "evaluation", "roc_curves.png"))
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    "Plots and saves the top N most important features from an XGBoost model."
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"XGBoost - Top {top_n} Features")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "..", "evaluation", "feature_importance.png"))
    plt.show()

def export_model_artifacts(xgb_model, lr_model, scaler, feature_names, output_dir=MODEL_DIR):
    "Saves the trained models, scaler, and feature names to disk."
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(xgb_model, f"{output_dir}/churn_model_xgb.pkl")
    joblib.dump(lr_model, f"{output_dir}/churn_model_lr.pkl")
    joblib.dump(scaler, f"{output_dir}/churn_scaler.pkl")
    joblib.dump(feature_names, f"{output_dir}/churn_features.pkl")
    print(f"Artifacts saved to {output_dir}/")

def predict_churn(customer_data, model_dir=MODEL_DIR):
    "Loads the model artifacts and returns the predicted churn probability, risk tier, and top risk factors."
    if not _model_cache:
        _model_cache["model"] = joblib.load(f"{model_dir}/churn_model_xgb.pkl")
        _model_cache["scaler"] = joblib.load(f"{model_dir}/churn_scaler.pkl")
        _model_cache["features"] = joblib.load(f"{model_dir}/churn_features.pkl")
        
    model = _model_cache["model"]
    scaler = _model_cache["scaler"]
    features = _model_cache["features"]
    
    row = pd.DataFrame([{f: customer_data.get(f, 0) for f in features}])
    prob = float(model.predict_proba(scaler.transform(row))[0][1])
    tier = "high" if prob >= 0.70 else "medium" if prob >= 0.40 else "low"
    
    top_idx = np.argsort(model.feature_importances_)[::-1][:3]
    top_factors = [features[i] for i in top_idx]
    
    return {
        "churn_probability": round(prob, 4),
        "risk_tier": tier,
        "top_risk_factors": top_factors,
    }

if __name__ == "__main__":
    clean_file = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_datafile.csv")
    print("--- Starting Model Building ---")
    
    if not os.path.exists(clean_file):
        print(f"Error: {clean_file} not found. Please run data_processing.py first.")
    else:
        df_clean = pd.read_csv(clean_file)
        
        # EDA & Feature Engineering
        compute_feature_associations(df_clean)
        plot_eda_charts(df_clean)
        df_engineered = engineer_features(df_clean)
        
        # Train / Test Split
        X_train, X_test, y_train, y_test, scaler = prepare_train_test(df_engineered)
        
        # Train Models
        lr_model = train_logistic(X_train, y_train)
        xgb_model = train_xgboost(X_train, y_train)
        
        # Evaluate Models
        lr_proba = evaluate_model("Logistic Regression", lr_model, X_test, y_test)
        xgb_proba = evaluate_model("XGBoost", xgb_model, X_test, y_test)
        
        # Plots
        plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost")
        plot_roc_curves({"Logistic Regression": lr_proba, "XGBoost": xgb_proba}, y_test)
        plot_feature_importance(xgb_model, X_train.columns.tolist())
        
        # Export Artifacts
        export_model_artifacts(xgb_model, lr_model, scaler, X_train.columns.tolist())
        print("\n--- Model Building Complete ---")
