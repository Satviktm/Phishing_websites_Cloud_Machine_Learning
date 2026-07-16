import joblib
import json
import time
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from extract_features import extract_features

app = FastAPI(title="Phishing Detection Prototype")

# ---- Load shared preprocessing artefacts ----
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
with open("selected_features.json") as f:
    feature_order = json.load(f)

# ---- Model routing config ----
model_configs = {
    "svm": {"file": "svm_model.pkl", "uses_pca": True,  "label": "SVM (LinearSVC)", "type": "Linear (margin-based)"},
    "lr":  {"file": "lr_model.pkl",  "uses_pca": True,  "label": "Logistic Regression", "type": "Linear (probabilistic)"},
    "xgb": {"file": "xgb_model.pkl", "uses_pca": False, "label": "XGBoost", "type": "Ensemble (boosted trees)"},
    "rf":  {"file": "rf_model.pkl",  "uses_pca": False, "label": "Random Forest", "type": "Ensemble (bagged trees)"},
}

models = {}
for key, cfg in model_configs.items():
    try:
        models[key] = joblib.load(cfg["file"])
        print(f"Loaded {key}")
    except FileNotFoundError:
        print(f"WARNING: {cfg['file']} not found — {key} will be unavailable")

# ---- Warm up loaded models (reduces cold-start latency on first real request) ----
_dummy_pca = np.zeros((1, pca.n_components_))
_dummy_scaled = np.zeros((1, len(feature_order)))
for key, cfg in model_configs.items():
    if key in models:
        try:
            dummy = _dummy_pca if cfg["uses_pca"] else _dummy_scaled
            models[key].predict(dummy)
        except Exception:
            pass
print("Models warmed up.")

# ---- Static reference numbers from each teammate's final evaluation ----
MODEL_METRICS = {
    "svm": {"name": "SVM (LinearSVC)", "accuracy": 0.9997, "precision": 0.9999,
            "recall": 0.9995, "f1": 0.9997, "auc": 0.999999, "latency_ms": 0.184},
    "lr":  {"name": "Logistic Regression", "accuracy": 0.999547, "precision": 0.999692,
            "recall": 0.999230, "f1": 0.999461, "auc": 0.999995, "latency_ms": 0.145},
    "xgb": {"name": "XGBoost", "accuracy": 0.999957, "precision": 1.000000,
            "recall": 0.999897, "f1": 0.999949, "auc": 0.999949, "latency_ms": 0.452},
    "rf":  {"name": "Random Forest", "accuracy": 1.0000, "precision": 1.0000,
            "recall": 0.9999, "f1": 0.9999, "auc": 1.0000, "latency_ms": 0.0037},
}

FEATURE_IMPORTANCE = {
    "svm": {"LargestLineLength": 0.1222, "NoOfExternalRef": 0.0799, "NoOfLettersInURL": 0.0754},
    "lr":  {"IsHTTPS": 0.1106, "HasSocialNet": 0.0933, "URLSimilarityIndex": 0.0688},
    "rf":  {"URLSimilarityIndex": 0.2370, "NoOfExternalRef": 0.1912, "NoOfSelfRef": 0.1183},
    "xgb": {"URLSimilarityIndex": 0.9858, "LineOfCode": 0.0113, "IsHTTPS": 0.0029},
}
KEY_FEATURES_TO_HIGHLIGHT = ['URLSimilarityIndex', 'NoOfSelfRef', 'IsHTTPS', 'HasDescription']

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def homepage():
    with open("static/index.html") as f:
        return f.read()


@app.get("/metrics")
def get_metrics():
    return MODEL_METRICS


@app.get("/feature_importance")
def get_feature_importance():
    return FEATURE_IMPORTANCE


def get_confidence(model, X_input):
    """
    Returns a 0-100 confidence score.
    Uses predict_proba where available (LR, RF, XGBoost).
    LinearSVC has no predict_proba by default, so its decision_function
    (distance from the margin) is squashed into a pseudo-confidence via a sigmoid.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        return round(float(max(proba)) * 100, 2)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_input)[0]
        pseudo_conf = 1 / (1 + np.exp(-abs(score)))
        return round(float(pseudo_conf) * 100, 2)
    return None


def get_top_factor(key):
    """Returns the single most influential feature for models with known importance data."""
    if key in FEATURE_IMPORTANCE:
        top_feature = max(FEATURE_IMPORTANCE[key], key=FEATURE_IMPORTANCE[key].get)
        importance_pct = FEATURE_IMPORTANCE[key][top_feature] * 100
        return f"{top_feature} ({importance_pct:.1f}% importance)"
    return None


class URLInput(BaseModel):
    url: str


@app.post("/predict_url")
def predict_url(data: URLInput):
    t_extract_start = time.time()
    raw_features = extract_features(data.url)
    t_extract = (time.time() - t_extract_start) * 1000

    fetch_warning = None
    if '_fetch_error' in raw_features:
        fetch_warning = "Could not fully analyze page content; some features used default values."

    values = [raw_features.get(name, 0) for name in feature_order]
    X = np.array([values])
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    results = []
    for key, cfg in model_configs.items():
        if key not in models:
            results.append({"model": key, "label": cfg["label"], "status": "unavailable"})
            continue

        t0 = time.time()
        X_input = X_pca if cfg["uses_pca"] else X_scaled
        model = models[key]
        prediction = model.predict(X_input)[0]
        confidence = get_confidence(model, X_input)
        latency_ms = (time.time() - t0) * 1000

        results.append({
            "model": key,
            "label": cfg["label"],
            "type": cfg["type"],
            "prediction": "Phishing" if prediction == 0 else "Legitimate",
            "raw_label": int(prediction),
            "confidence": confidence,
            "top_factor": get_top_factor(key),
            "latency_ms": round(latency_ms, 4),
            "status": "ok"
        })

    return {
        "url": data.url,
        "extraction_time_ms": round(t_extract, 2),
        "results": results,
        "fetch_warning": fetch_warning,
        "key_evidence": {k: raw_features.get(k, 0) for k in KEY_FEATURES_TO_HIGHLIGHT}
    }