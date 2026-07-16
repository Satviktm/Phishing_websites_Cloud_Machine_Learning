# Phishing URL Detection — Cloud ML Prototype

Live web prototype comparing 4 machine learning models (SVM, Logistic Regression,
Random Forest, XGBoost) trained on the PhiUSIIL Phishing URL Dataset, deployed on
AWS EC2/Cloud9 via FastAPI.

## Files
- `app.py` — FastAPI backend, loads all 4 models and serves predictions
- `extract_features.py` — live URL feature extraction (fetches page, computes the 30 features)
- `static/index.html` — front-end dashboard
- `static/images/` — evaluation visualisations (confusion matrices, metrics comparison)
- `*.pkl` — trained model files and shared preprocessing artefacts (scaler, PCA)
- `selected_features.json` — the 30 official selected features, in required order

## Running locally
## Team
Cloud Machine Learning — Group 08
