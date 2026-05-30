import json
import os
import datetime
import joblib
from .model_building import predict_churn as ml_predict_churn

MOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")
CUSTOMERS_FILE = os.path.join(MOCK_DATA_DIR, "customers.json")
OFFERS_FILE = os.path.join(MOCK_DATA_DIR, "offers.json")
INTERACTIONS_FILE = os.path.join(MOCK_DATA_DIR, "interactions_log.jsonl")
ESCALATIONS_FILE = os.path.join(MOCK_DATA_DIR, "escalations_log.jsonl")

_execution_trace = []

def get_and_clear_trace():
    global _execution_trace
    trace = list(_execution_trace)
    _execution_trace.clear()
    return trace

def _load_json(filepath: str) -> list:
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def _append_jsonl(filepath: str, data: dict):
    with open(filepath, 'a') as f:
        f.write(json.dumps(data) + "\n")

# --- TOOL 1: Lookup Customer ---
def lookup_customer(customer_id: str) -> dict:
    "Retrieves the raw, human-readable profile of a customer by their ID (e.g., CUST-001) from the mock database."
    customer_id = str(customer_id).upper().strip()
    _execution_trace.append({"tool": "lookup_customer", "customer_id": customer_id})
    customers = _load_json(CUSTOMERS_FILE)
    for c in customers:
        if c.get("customer_id") == customer_id:
            return c
    return {"error": f"Customer ID {customer_id} not found."}

# --- TOOL 2: Predict Churn ---
def _translate_to_model_features(raw_data: dict) -> dict:
    "Translates human-readable customer data to the numeric format expected by the predict_churn model."
    features = {}
    features["age"] = float(raw_data.get("age", 35))
    features["tenure_months"] = float(raw_data.get("tenure_months", 1))
    features["monthly_charges"] = float(raw_data.get("monthly_charges", 50.0))
    features["total_charges"] = float(raw_data.get("total_charges", 50.0))
    features["avg_monthly_gb_used"] = float(raw_data.get("avg_monthly_gb_used", 10.0))
    features["avg_monthly_minutes"] = float(raw_data.get("avg_monthly_minutes", 100.0))
    features["num_support_tickets"] = float(raw_data.get("num_support_tickets", 0))
    features["num_additional_services"] = float(raw_data.get("num_additional_services", 0))
    features["satisfaction_score"] = float(raw_data.get("satisfaction_score", 5))
    features["days_since_last_interaction"] = float(raw_data.get("days_since_last_interaction", 10))
    
    # Derived features
    features["charges_per_month_ratio"] = features["total_charges"] / max(features["tenure_months"], 1)
    
    # Categoricals mapping
    contract = str(raw_data.get("contract", "Month-to-month")).strip()
    cont_val = {"Month-to-month": 0, "One year": 1, "Two year": 2}.get(contract, 0)
    features["contract_type"] = cont_val
    
    sat_norm = max(0, 10 - features["satisfaction_score"]) / 10.0
    ten_norm = max(0, 72 - features["tenure_months"]) / 72.0
    cont_norm = (2 - cont_val) / 2.0
    features["risk_score_composite"] = round((0.40 * sat_norm) + (0.35 * cont_norm) + (0.25 * ten_norm), 4)
    
    features["phone_service"] = 1 if str(raw_data.get("phone_service", "")).lower() in ["yes", "1"] else 0
    
    gender = str(raw_data.get("gender", "Unknown")).strip().capitalize()
    features["gender_Female"] = 1 if gender == "Female" else 0
    features["gender_Male"] = 1 if gender == "Male" else 0
    
    internet = str(raw_data.get("internet_service", "Unknown")).strip()
    features["internet_DSL"] = 1 if internet == "DSL" else 0
    features["internet_Fiber optic"] = 1 if internet == "Fiber optic" else 0
    features["internet_No internet"] = 1 if internet == "No internet" else 0
    
    payment = str(raw_data.get("payment_method", "Unknown")).strip().capitalize()
    features["payment_Bank transfer"] = 1 if payment == "Bank transfer" else 0
    features["payment_Credit card"] = 1 if payment == "Credit card" else 0
    features["payment_Electronic check"] = 1 if payment == "Electronic check" else 0
    features["payment_Mailed check"] = 1 if payment == "Mailed check" else 0
    
    # Adding flags
    features["has_any_tickets"] = 1 if features["num_support_tickets"] > 0 else 0
    features["is_high_ticket"] = 1 if features["num_support_tickets"] >= 4 else 0
    features["is_dissatisfied"] = 1 if features["satisfaction_score"] <= 4 else 0
    features["has_any_addons"] = 1 if features["num_additional_services"] > 0 else 0
    features["max_addons"] = 1 if features["num_additional_services"] == 5 else 0
    features["long_since_interaction"] = 1 if features["days_since_last_interaction"] > 90 else 0

    features["conflict_phone_minutes"] = 1 if (features["phone_service"] == 0 and features["avg_monthly_minutes"] > 0) else 0
    features["conflict_internet_gb"] = 1 if (features["internet_No internet"] == 1 and features["avg_monthly_gb_used"] > 0) else 0
    features["conflict_no_service_addons"] = 1 if (features["internet_No internet"] == 1 and features["phone_service"] == 0 and features["num_additional_services"] > 0) else 0
    features["conflict_internet_addons"] = 1 if (features["internet_No internet"] == 1 and features["num_additional_services"] > 0) else 0
    features["conflict_contract_tenure"] = 1 if (features["contract_type"] == 2 and features["tenure_months"] < 3) else 0
    features["conflict_satisfaction_tickets"] = 1 if (features["satisfaction_score"] >= 8 and features["num_support_tickets"] >= 4) else 0

    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "churn_features.pkl")
        model_features = joblib.load(model_path)
        for f in model_features:
            if f not in features:
                features[f] = 0
        final_features = {f: features.get(f, 0) for f in model_features}
    except Exception:
        final_features = features
        
    return final_features

def predict_customer_churn(customer_data: dict) -> dict:
    "Accepts a customer profile dictionary (features), translates it to ML format, and returns churn probability, risk tier, and top risk factors. You MUST lookup the customer first to get this dictionary."
    _execution_trace.append({"tool": "predict_customer_churn"})
    try:
        model_features = _translate_to_model_features(customer_data)
        result = ml_predict_churn(model_features)
        return result
    except Exception as e:
        return {"error": f"Failed to predict churn: {str(e)}"}

# --- TOOL 3: Get Retention Offers ---
def get_retention_offers(risk_tier: str, contract_type: str) -> list:
    "Returns a list of retention offers filtered by the customer's risk tier ('high', 'medium', 'low') and contract type ('Month-to-month', etc)."
    _execution_trace.append({"tool": "get_retention_offers", "risk_tier": risk_tier, "contract_type": contract_type})
    offers = _load_json(OFFERS_FILE)
    valid_offers = []
    for offer in offers:
        if risk_tier in offer.get("valid_risk_tiers", []) and contract_type in offer.get("valid_contract_types", []):
            valid_offers.append(offer)
    return valid_offers

# --- TOOL 4: Log Interaction ---
def log_interaction(customer_id: str, outcome: str, notes: str) -> str:
    "Records the outcome of a retention conversation in a persistent log file."
    customer_id = str(customer_id).upper().strip()
    _execution_trace.append({"tool": "log_interaction", "customer_id": customer_id})
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "customer_id": customer_id,
        "outcome": outcome,
        "notes": notes
    }
    _append_jsonl(INTERACTIONS_FILE, log_entry)
    return f"Successfully logged interaction for {customer_id}."

# --- TOOL 5: Escalate to Supervisor ---
def escalate_to_supervisor(customer_id: str, summary: str, reason: str) -> str:
    "Transfers the case to a human supervisor with a context summary for situations the agent should not handle alone."
    customer_id = str(customer_id).upper().strip()
    _execution_trace.append({"tool": "escalate_to_supervisor", "customer_id": customer_id})
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "customer_id": customer_id,
        "summary": summary,
        "reason": reason,
        "status": "pending_human_review"
    }
    _append_jsonl(ESCALATIONS_FILE, log_entry)
    return f"Case for {customer_id} successfully escalated to supervisor. A human representative will take over."