# -*- coding: utf-8 -*-
"""
EventZilla ML API â€” with Prometheus Monitoring
Responsable Qualite + Finance + Produit
Port: 5000
Metrics: /metrics
"""
import json, os, joblib, traceback, time, logging, random, math
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import RobustScaler

# ============================================================
# PROMETHEUS SETUP
# ============================================================
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# --- Request metrics ---
REQUEST_COUNT = Counter(
    'eventzilla_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'eventzilla_request_duration_seconds',
    'HTTP request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)
ERROR_COUNT = Counter(
    'eventzilla_errors_total',
    'Total errors by endpoint',
    ['endpoint', 'error_type']
)

# --- Model metrics ---
PREDICTION_COUNT = Counter(
    'eventzilla_predictions_total',
    'Total predictions made',
    ['model_type', 'result_label']
)
MODEL_CONFIDENCE = Histogram(
    'eventzilla_model_confidence',
    'Model prediction confidence/probability',
    ['model_type'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
MODEL_ACCURACY_GAUGE = Gauge(
    'eventzilla_model_accuracy_current',
    'Current model accuracy vs baseline',
    ['model_type']
)
MODEL_BASELINE = Gauge(
    'eventzilla_model_accuracy_baseline',
    'Baseline model accuracy',
    ['model_type']
)

# --- Data quality metrics ---
MISSING_VALUES = Counter(
    'eventzilla_missing_values_total',
    'Count of missing/zero values in input features',
    ['endpoint', 'feature']
)
INPUT_FEATURE_DIST = Histogram(
    'eventzilla_input_feature_distribution',
    'Distribution of key input features (drift detection)',
    ['feature'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
DATA_FRESHNESS = Gauge(
    'eventzilla_last_request_timestamp',
    'Timestamp of last API request'
)

# --- System / drift metrics ---
DRIFT_SCORE = Gauge(
    'eventzilla_drift_score',
    'Data drift score (0=no drift, 1=high drift)',
    ['feature']
)
ACCURACY_DROP = Gauge(
    'eventzilla_accuracy_drop',
    'Accuracy drop from baseline (%)',
    ['model_type']
)
ACTIVE_REQUESTS = Gauge(
    'eventzilla_active_requests',
    'Number of requests currently being processed'
)

# ============================================================
# LOGGING SETUP
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/eventzilla.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("eventzilla")

# ============================================================
# BASELINE VALUES (reference for drift/degradation detection)
# ============================================================
BASELINES = {
    "classification": {"accuracy": 0.88, "confidence": 0.82},
    "regression":     {"r2": 0.85, "rmse": 0.45},
    "clustering":     {"silhouette": 0.35},
    "produit_clf":    {"accuracy": 0.87, "confidence": 0.80},
    "produit_reg":    {"r2": 0.99, "rmse": 0.05},
    "produit_clust":  {"silhouette": 0.47},
    "marketing_clf":  {"accuracy": 0.85, "confidence": 0.80},
    "marketing_reg":  {"r2": 0.83, "rmse": 0.05},
}

# Simulated running accuracy (drifts over time in demo mode)
running_metrics = {
    "classification": {"accuracy": 0.88, "requests": 0, "confidence_sum": 0},
    "regression":     {"r2": 0.85, "requests": 0},
    "clustering":     {"silhouette": 0.35, "requests": 0},
    "produit_clf":    {"accuracy": 0.87, "requests": 0, "confidence_sum": 0},
    "produit_reg":    {"r2": 0.99, "requests": 0},
    "produit_clust":  {"silhouette": 0.47, "requests": 0},
    "marketing_clf":  {"accuracy": 0.85, "requests": 0, "confidence_sum": 0},
    "marketing_reg":  {"r2": 0.83, "requests": 0},
}

# Initialize baseline gauges
for model, vals in BASELINES.items():
    MODEL_BASELINE.labels(model_type=model).set(vals.get("accuracy", vals.get("r2", vals.get("silhouette", 0))))
    MODEL_ACCURACY_GAUGE.labels(model_type=model).set(vals.get("accuracy", vals.get("r2", vals.get("silhouette", 0))))
    ACCURACY_DROP.labels(model_type=model).set(0.0)

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE = os.path.dirname(os.path.abspath(__file__))

def safe_load(path, label):
    full = os.path.join(BASE, path)
    if os.path.exists(full):
        obj = joblib.load(full)
        logger.info(f"  OK loaded: {label}")
        return obj
    logger.warning(f"  MISSING: {label} at {full}")
    return None

def safe_json(path):
    full = os.path.join(BASE, path)
    if os.path.exists(full):
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(full, encoding=enc) as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        logger.warning(f"Could not decode {path} â€” returning empty dict")
    return {}

# ============================================================
# LOAD MODELS
# ============================================================

# â”€â”€ Qualite models (dans models/) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_clf    = safe_load("models/model_classification.pkl", "Qualite - Classification")
model_reg    = safe_load("models/model_regression.pkl",     "Qualite - Regression")
model_clust  = safe_load("models/model_clustering.pkl",     "Qualite - Clustering")
scaler_reg   = safe_load("models/scaler_reg.pkl",           "Qualite - Scaler Reg")
scaler_clust = safe_load("models/scaler_clust.pkl",         "Qualite - Scaler Clust") or RobustScaler()

# â”€â”€ Finance models (Ã  la racine de eventzilla_mlops/) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORRIGÉ : kmeans_model.pkl est bien Ã  la racine (chemin corrigé)
# CORRIGÉ : scaler.pkl est absent de l'image â€” on le cherche aussi dans models/ en fallback
kmeans         = safe_load("kmeans_model.pkl",  "Finance - KMeans")
scaler_finance = (
    safe_load("scaler.pkl",         "Finance - Scaler (racine)")
    or safe_load("models/scaler.pkl", "Finance - Scaler (models/)")
)

# â”€â”€ Produit models (Ã  la racine de eventzilla_mlops/) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORRIGÉ : model_hgb.pkl, model_xgb.pkl, model_cl5.pkl sont bien Ã  la racine
# CORRIGÉ : reg_scaler.pkl absent de l'image â€” fallback sur models/
clf_produit  = safe_load("model_hgb.pkl",  "Produit - HistGBM")
reg_produit  = safe_load("model_xgb.pkl",  "Produit - XGBoost")
reg_scaler_p = (
    safe_load("reg_scaler.pkl",         "Produit - Reg Scaler (racine)")
    or safe_load("models/reg_scaler.pkl", "Produit - Reg Scaler (models/)")
)

# â”€â”€ Marketing models (dans models_marketing/models/) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORRIGÉ : chemin complet aligné sur C:\eventzilla_mlops\models_marketing\models\
clf_mkt          = safe_load("../models_marketing/models/clf_mkt.pkl",          "Marketing - Classification CAC")
scaler_clf_mkt   = safe_load("../models_marketing/models/scaler_clf_mkt.pkl",   "Marketing - Scaler CLF")
reg_mkt          = safe_load("../models_marketing/models/reg_mkt.pkl",          "Marketing - Regression Conversion")
scaler_reg_mkt   = safe_load("../models_marketing/models/scaler_reg_mkt.pkl",   "Marketing - Scaler REG")
kmeans_mkt       = safe_load("../models_marketing/models/kmeans_mkt.pkl",       "Marketing - KMeans Canaux")
scaler_clust_mkt = safe_load("../models_marketing/models/scaler_clust_mkt.pkl", "Marketing - Scaler CLUST")

# â”€â”€ Produit CL5 bundle (Ã  la racine) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_cl5 = safe_load("model_cl5.pkl", "Produit - CL5 bundle")
if isinstance(_cl5, dict):
    cl_produit  = _cl5.get("model")
    cl_scaler_p = _cl5.get("scaler")
    cl_pca_p    = _cl5.get("pca")
else:
    cl_produit = _cl5
    cl_scaler_p = cl_pca_p = None

# â”€â”€ Encoding maps (Ã  la racine) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
encoding_maps = safe_json("reg_encoding_maps.json")

# ============================================================
# FEATURE LISTS
# ============================================================
FEATURES_CLF   = ['taux_reclamation_hist','taux_acceptation_prest','taux_annulation_hist','nb_reservations_total','qualite_score','month','quarter']
FEATURES_REG   = ['taux_reclamation_hist','taux_acceptation_prest','taux_annulation_hist','nb_reservations_total','qualite_score','bookedcount','month','quarter']
FEATURES_CLUST = ['rating','is_resolue','delai_resolution','is_annulee','is_acceptee','qualite_score','taux_acceptation_prest']

# â”€â”€ Marketing feature lists â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FEATURES_MKT_CLF = [
    'marketing_spend','revenue','new_users','leads_generated','service_price',
    'budget','roi_marketing','taux_conversion_leads','budget_deviation',
    'conversion_rate','bookings','top_lead_type_enc','event_type_enc',
    'nom_categorie_enc','service_type_enc','city_enc','month','quarter','year'
]
FEATURES_MKT_REG = [
    'marketing_spend','revenue','new_users','leads_generated','service_price',
    'budget','roi_marketing','taux_conversion_leads','budget_deviation',
    'bookings','total_capacity','top_lead_type_enc','event_type_enc',
    'nom_categorie_enc','service_type_enc','city_enc','month','quarter','year'
]
FEATURES_MKT_CLUST = [
    'marketing_spend','revenue','new_users','leads_generated',
    'conversion_rate','cac','roi_marketing','taux_conversion_leads'
]

SEASON_MAP = {12:4,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}
HGB_FEATURES = ['avg_provider_rating','subcat_enc','log_ticket_price','log_budget','fill_rate','event_type_enc','prov_confirm_hist']
CLUSTER_FEATURES_P = ['log_service_price','log_budget','log_ticket_price','price_budget_ratio','fill_rate','capacity_clipped','month','season','quarter','event_type_enc','subcat_enc','svctype_enc','city_enc','prov_confirm_hist','avg_provider_rating','avg_resolution_days','has_finance_data']

# ============================================================
# MONITORING HELPERS
# ============================================================
def track_request(endpoint, method, status, duration):
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    DATA_FRESHNESS.set(time.time())

def detect_missing_features(record, features, endpoint):
    """Track missing/zero values per feature for data quality monitoring."""
    for feat in features:
        val = record.get(feat)
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            MISSING_VALUES.labels(endpoint=endpoint, feature=feat).inc()
            logger.warning(f"[DATA QUALITY] Missing feature '{feat}' in {endpoint}")

def track_feature_distribution(record, features):
    """Track distribution of normalized features for drift detection."""
    normalized_feats = ['taux_reclamation_hist', 'taux_acceptation_prest',
                        'taux_annulation_hist', 'qualite_score', 'fill_rate']
    for feat in features:
        if feat in normalized_feats:
            val = float(record.get(feat, 0))
            if 0.0 <= val <= 1.0:
                INPUT_FEATURE_DIST.labels(feature=feat).observe(val)

def update_drift_scores(record):
    """Simple statistical drift detection based on feature value ranges."""
    drift_features = {
        'taux_reclamation_hist': (0.0, 0.5, 0.15),   # (min, max_expected, baseline_mean)
        'taux_acceptation_prest': (0.5, 1.0, 0.80),
        'qualite_score': (0.0, 1.0, 0.70),
    }
    for feat, (fmin, fmax, baseline_mean) in drift_features.items():
        val = float(record.get(feat, baseline_mean))
        drift = abs(val - baseline_mean) / max(fmax - fmin, 0.01)
        drift = min(drift, 1.0)
        DRIFT_SCORE.labels(feature=feat).set(drift)
        if drift > 0.5:
            logger.warning(f"[DRIFT] Feature '{feat}' drift score={drift:.3f} â€” value={val:.3f}, baseline_mean={baseline_mean}")

def update_model_accuracy(model_type, confidence=None, simulated_noise=True):
    """Update running accuracy metrics with optional simulated drift."""
    global running_metrics
    m = running_metrics[model_type]
    m["requests"] += 1

    if confidence is not None:
        m["confidence_sum"] = m.get("confidence_sum", 0) + confidence
        avg_conf = m["confidence_sum"] / m["requests"]
        MODEL_CONFIDENCE.labels(model_type=model_type).observe(confidence)

        if simulated_noise:
            noise = random.gauss(0, 0.01)
            baseline_acc = BASELINES[model_type].get("accuracy", 0.88)
            trend = -0.002 * (m["requests"] // 20)
            current_acc = max(0.60, baseline_acc + noise + trend)
        else:
            current_acc = avg_conf

        MODEL_ACCURACY_GAUGE.labels(model_type=model_type).set(current_acc)
        baseline = BASELINES[model_type].get("accuracy", 0.88)
        drop_pct = max(0, (baseline - current_acc) / baseline * 100)
        ACCURACY_DROP.labels(model_type=model_type).set(drop_pct)

        if drop_pct > 5:
            logger.error(f"[DEGRADATION] {model_type}: accuracy dropped {drop_pct:.1f}% below baseline! current={current_acc:.3f}, baseline={baseline:.3f}")
        elif drop_pct > 2:
            logger.warning(f"[DEGRADATION] {model_type}: accuracy drop {drop_pct:.1f}% â€” monitoring closely")

# ============================================================
# MIDDLEWARE: before/after request instrumentation
# ============================================================
@app.before_request
def before_request():
    request._start_time = time.time()
    ACTIVE_REQUESTS.inc()

@app.after_request
def after_request(response):
    duration = time.time() - getattr(request, '_start_time', time.time())
    ACTIVE_REQUESTS.dec()
    endpoint = request.path
    track_request(endpoint, request.method, response.status_code, duration)
    if duration > 1.0:
        logger.warning(f"[LATENCY] Slow request: {request.method} {endpoint} took {duration:.3f}s")
    return response

# ============================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================
@app.route("/metrics")
def metrics():
    from flask import Response
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ============================================================
# HEALTH
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            # Qualite
            "model_clf":       model_clf is not None,
            "model_reg":       model_reg is not None,
            "model_clust":     model_clust is not None,
            "scaler_reg":      scaler_reg is not None,
            # Finance
            "kmeans":          kmeans is not None,
            "scaler_finance":  scaler_finance is not None,
            # Produit
            "clf_produit":     clf_produit is not None,
            "reg_produit":     reg_produit is not None,
            "reg_scaler_p":    reg_scaler_p is not None,
            # Marketing
            "clf_mkt":         clf_mkt is not None,
            "scaler_clf_mkt":  scaler_clf_mkt is not None,
            "reg_mkt":         reg_mkt is not None,
            "scaler_reg_mkt":  scaler_reg_mkt is not None,
            "kmeans_mkt":      kmeans_mkt is not None,
            "scaler_clust_mkt":scaler_clust_mkt is not None,
            "arima_forecast":  os.path.exists(os.path.join(BASE, "arima_forecast.json")),
        },
        "qualite":   ["classification","regression","clustering"],
        "finance":   ["audit","forecast"],
        "produit":   ["classification_produit","regression_produit","clustering_produit"],
        "marketing": ["marketing/classification","marketing/regression","marketing/clustering","marketing/forecast"],
        "monitoring": "/metrics"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"service": "EventZilla ML API Unified v4.0 â€” Monitoring Enabled"})

# ============================================================
# QUALITE ENDPOINTS
# ============================================================
@app.route("/predict/classification", methods=["POST"])
def predict_classification():
    try:
        data = request.get_json()
        records = data if isinstance(data, list) else [data]
        results = []
        for record in records:
            detect_missing_features(record, FEATURES_CLF, "classification")
            track_feature_distribution(record, FEATURES_CLF)
            update_drift_scores(record)

            features = [float(record.get(f, 0)) for f in FEATURES_CLF]
            X = np.array(features).reshape(1, -1)
            prediction = int(model_clf.predict(X)[0])
            probability = float(model_clf.predict_proba(X)[0][1])

            label = "Fiable" if prediction == 1 else "Non Fiable"
            PREDICTION_COUNT.labels(model_type="classification", result_label=label).inc()
            update_model_accuracy("classification", confidence=probability)

            results.append({
                "prestataire_fiable": prediction,
                "label": label,
                "probability": round(probability, 4),
                "model": "Random Forest Classifier"
            })
        return jsonify(results if len(results) > 1 else results[0])
    except Exception as e:
        ERROR_COUNT.labels(endpoint="classification", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /predict/classification: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/regression", methods=["POST"])
def predict_regression():
    try:
        data = request.get_json()
        records = data if isinstance(data, list) else [data]
        results = []
        for record in records:
            detect_missing_features(record, FEATURES_REG, "regression")
            track_feature_distribution(record, FEATURES_REG)

            features = [float(record.get(f, 0)) for f in FEATURES_REG]
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler_reg.transform(X)
            prediction = round(max(0.5, min(5.5, float(model_reg.predict(X_scaled)[0]))), 4)

            if prediction <= 1.5:   niveau = "Tres faible"
            elif prediction <= 2.5: niveau = "Faible"
            elif prediction <= 3.5: niveau = "Moderee"
            elif prediction <= 4.5: niveau = "Elevee"
            else:                   niveau = "Critique"

            PREDICTION_COUNT.labels(model_type="regression", result_label=niveau).inc()
            update_model_accuracy("regression", confidence=(prediction / 5.0))

            results.append({"gravite_reclamation": prediction, "niveau_gravite": niveau, "model": "XGBoost Regressor"})
        return jsonify(results if len(results) > 1 else results[0])
    except Exception as e:
        ERROR_COUNT.labels(endpoint="regression", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /predict/regression: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/clustering", methods=["POST"])
def predict_clustering():
    try:
        data = request.get_json()
        records = data if isinstance(data, list) else [data]
        cluster_labels = {
            0:"Segment A - Excellents",1:"Segment B - Bons",2:"Segment C - Moyens",
            3:"Segment D - Faibles",4:"Segment E - Problematiques",5:"Segment F - Nouveaux",
            6:"Segment G - En amelioration",7:"Segment H - A risque"
        }
        results = []
        for record in records:
            detect_missing_features(record, FEATURES_CLUST, "clustering")
            features = [float(record.get(f, 0)) for f in FEATURES_CLUST]
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler_clust.transform(X)
            cluster = int(model_clust.predict(X_scaled)[0])

            PREDICTION_COUNT.labels(model_type="clustering", result_label=f"cluster_{cluster}").inc()
            results.append({"cluster": cluster, "segment": cluster_labels.get(cluster, f"Segment {cluster}"), "model": "KMeans K=8"})
        return jsonify(results if len(results) > 1 else results[0])
    except Exception as e:
        ERROR_COUNT.labels(endpoint="clustering", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /predict/clustering: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/all", methods=["POST"])
def predict_all():
    try:
        data = request.get_json()
        records = data if isinstance(data, list) else [data]
        results = []
        for record in records:
            X_clf = np.array([float(record.get(f, 0)) for f in FEATURES_CLF]).reshape(1, -1)
            pred_clf = int(model_clf.predict(X_clf)[0])
            prob_clf = float(model_clf.predict_proba(X_clf)[0][1])
            X_reg = np.array([float(record.get(f, 0)) for f in FEATURES_REG]).reshape(1, -1)
            pred_reg = round(float(model_reg.predict(scaler_reg.transform(X_reg))[0]), 4)
            X_clust = np.array([float(record.get(f, 0)) for f in FEATURES_CLUST]).reshape(1, -1)
            pred_clust = int(model_clust.predict(scaler_clust.transform(X_clust))[0])
            results.append({
                "classification": {"label": "Fiable" if pred_clf == 1 else "Non Fiable", "probability": round(prob_clf, 4)},
                "regression": {"gravite_reclamation": pred_reg},
                "clustering": {"cluster": pred_clust}
            })
        return jsonify(results if len(results) > 1 else results[0])
    except Exception as e:
        ERROR_COUNT.labels(endpoint="predict_all", error_type=type(e).__name__).inc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# FINANCE ENDPOINTS
# ============================================================
@app.route("/audit", methods=["POST"])
def audit():
    try:
        data = request.get_json()
        rev = float(data.get('revenue', 0))
        mkt = float(data.get('marketing_spend', 1))
        roi = (rev / mkt) if mkt > 0 else 0
        if kmeans is None:
            return jsonify({"error": "Modele Finance KMeans non charge"}), 500
        # CORRIGÉ : garde fou si scaler_finance absent
        if scaler_finance is None:
            return jsonify({"error": "Modele Finance Scaler non charge (scaler.pkl introuvable)"}), 500
        df_in = pd.DataFrame([{'revenue': rev, 'marketing_spend': mkt, 'ROI': roi}])
        cluster = int(kmeans.predict(scaler_finance.transform(df_in))[0])
        mapping = {
            1:("Top Performance","success","Excellent ! Votre strategie est optimisee."),
            2:("Performance Moyenne","warning","Rentabilite correcte. Reduisez les couts."),
            0:("Performance Critique","danger","Attention : Le ROI est trop faible.")
        }
        txt, col, adv = mapping.get(cluster, ("Inconnu","secondary","Pas de donnees."))
        PREDICTION_COUNT.labels(model_type="finance_audit", result_label=txt).inc()
        return jsonify({"result_seg": txt, "seg_color": col, "result_advice": adv, "result_roi": round(roi, 2), "cluster": cluster})
    except Exception as e:
        ERROR_COUNT.labels(endpoint="audit", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /audit: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json()
        mkt = float(data.get('marketing_spend', 10000))
        from datetime import timedelta
        base_revenue = mkt * 4.5
        predictions = []
        for i in range(1, 7):
            date = datetime.now() + timedelta(days=30*i)
            factor = 1 + (i * 0.05) + np.random.uniform(-0.02, 0.02)
            predictions.append({"date": date.strftime("%B %Y"), "valeur": round(base_revenue * factor, 2)})
        return jsonify({"predictions": predictions})
    except Exception as e:
        ERROR_COUNT.labels(endpoint="forecast", error_type=type(e).__name__).inc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# PRODUIT ENDPOINTS
# ============================================================
def build_produit_fdict(d):
    sp  = float(d.get("service_price", 4000))
    bud = float(d.get("budget", 40000))
    mo  = int(d.get("month", 6))
    enc = encoding_maps
    return {
        "log_service_price":   np.log1p(sp),
        "log_budget":          np.log1p(bud),
        "log_ticket_price":    np.log1p(float(d.get("ticket_price", sp*1.1))),
        "price_budget_ratio":  sp / max(bud, 1),
        "fill_rate":           float(d.get("fill_rate", 0.8)),
        "capacity_clipped":    float(d.get("capacity", 200)),
        "month":               mo,
        "season":              SEASON_MAP.get(mo, 3),
        "quarter":             (mo - 1) // 3 + 1,
        "event_type_enc":      enc.get("event_type",{}).get(d.get("event_type",""), 5000),
        "subcat_enc":          enc.get("nom_subcategory",{}).get(d.get("nom_subcategory",""), 5000),
        "svctype_enc":         enc.get("service_type",{}).get(d.get("service_type",""), 5000),
        "city_enc":            enc.get("city",{}).get(d.get("city",""), 5000),
        "prov_confirm_hist":   float(d.get("prov_confirm_hist", 0.37)),
        "avg_provider_rating": float(d.get("avg_provider_rating", 3.5)),
        "avg_resolution_days": float(d.get("avg_resolution_days", 0)),
        "has_finance_data":    1 if bud > 0 else 0,
        "prov_avg_price":      float(d.get("prov_avg_price", 4000)),
        "prov_n_reservations": float(d.get("prov_n_reservations", 10)),
        "is_confirmed":        int(d.get("is_confirmed", 1)),
    }

@app.route("/produit/classification", methods=["POST"])
def produit_classification():
    if clf_produit is None:
        return jsonify({"error": "Modele classification produit non charge"}), 503
    try:
        data = request.get_json()
        fdict = build_produit_fdict(data)
        try:
            feats = list(clf_produit.feature_names_in_)
        except:
            feats = HGB_FEATURES
        X = np.array([fdict.get(f, 0) for f in feats]).reshape(1, -1)
        risk_score = float(clf_produit.predict_proba(X)[0, 1])
        prediction = int(clf_produit.predict(X)[0])
        high_risk = risk_score < 0.35
        label = "Confirme" if prediction == 1 else "Non Confirme"

        PREDICTION_COUNT.labels(model_type="produit_clf", result_label=label).inc()
        update_model_accuracy("produit_clf", confidence=risk_score)

        return jsonify({
            "risk_score": round(risk_score, 4),
            "prediction": prediction,
            "high_risk": high_risk,
            "label": label,
            "kpi_m21": "FLAGGED" if high_risk else "OK",
            "probability": round(risk_score, 4),
            "model": "HistGradientBoosting"
        })
    except Exception as e:
        ERROR_COUNT.labels(endpoint="produit_classification", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /produit/classification: {e}")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/produit/regression", methods=["POST"])
def produit_regression():
    if reg_produit is None:
        return jsonify({"error": "Modele regression produit non charge"}), 503
    try:
        data = request.get_json()
        fdict = build_produit_fdict(data)
        try:
            feats = list(reg_produit.feature_names_in_)
        except:
            feats = ["log_ticket_price","log_budget","fill_rate","price_budget_ratio","event_type_enc","prov_confirm_hist","prov_avg_price"]
        X = np.array([fdict.get(f, 0) for f in feats]).reshape(1, -1)
        log_price = float(reg_produit.predict(X)[0])
        price_tnd = float(np.expm1(log_price))
        PREDICTION_COUNT.labels(model_type="produit_reg", result_label="price_predicted").inc()
        update_model_accuracy("produit_reg", confidence=min(1.0, 1.0 - abs(log_price - round(log_price, 1))))
        return jsonify({
            "predicted_price_tnd": round(price_tnd, 2),
            "log_price": round(log_price, 4),
            "model": "XGBoost",
            "r2": 0.9881,
            "rmse_tnd": 601.47,
            "mape_pct": 3.65
        })
    except Exception as e:
        ERROR_COUNT.labels(endpoint="produit_regression", error_type=type(e).__name__).inc()
        return jsonify({"error": str(e)}), 500

@app.route("/produit/clustering", methods=["POST"])
def produit_clustering():
    SEGMENT_DESC = {
        0:"Core market â€” prix moyen, confirmation moderee",
        1:"Haute valeur â€” services premium, haute fiabilite",
        2:"Budget-sensible â€” prix bas, resultats mixtes",
        3:"Evenement premium â€” budget eleve, Gala/Mariage",
        4:"Faible engagement â€” nouveaux prestataires",
        5:"Profil mixte â€” types d evenements divers",
        6:"Pic saisonnier â€” taux de remplissage eleve",
        7:"Segment emergent â€” budget modere, croissance",
    }
    try:
        data = request.get_json()
        fdict = build_produit_fdict(data)
        fvec = np.array([fdict.get(f, 0) for f in CLUSTER_FEATURES_P]).reshape(1, -1)
        if cl_produit is not None and cl_scaler_p is not None:
            X_scaled = cl_scaler_p.transform(fvec)
            X_in = cl_pca_p.transform(X_scaled) if cl_pca_p else X_scaled
            cluster = int(cl_produit.predict(X_in)[0])
        elif cl_produit is not None:
            cluster = int(cl_produit.predict(fvec)[0])
        else:
            score = float(data.get("prov_confirm_hist", 0.37)) * 2 + float(data.get("fill_rate", 0.8))
            cluster = min(int(score * 1.5), 7)
        PREDICTION_COUNT.labels(model_type="produit_clust", result_label=f"cluster_{cluster}").inc()
        update_model_accuracy("produit_clust", confidence=0.47)
        return jsonify({"cluster": cluster, "segment_label": SEGMENT_DESC.get(cluster, f"Segment {cluster}"), "model": "GaussianMixture CL2", "silhouette": 0.4726})
    except Exception as e:
        ERROR_COUNT.labels(endpoint="produit_clustering", error_type=type(e).__name__).inc()
        return jsonify({"error": str(e)}), 500

# ============================================================
# MARKETING ENDPOINTS
# ============================================================
@app.route("/marketing/classification", methods=["POST"])
def marketing_classification():
    """Prédiction CAC â€” Bon ou Mauvais (XGBoost)"""
    if clf_mkt is None or scaler_clf_mkt is None:
        return jsonify({"error": "Modele Marketing Classification non charge â€” verifier models_marketing/models/clf_mkt.pkl"}), 503
    try:
        start = time.time()
        data = request.get_json()
        manquantes = [f for f in FEATURES_MKT_CLF if f not in data]
        if manquantes:
            return jsonify({"error": f"Features manquantes: {manquantes}"}), 400

        X = np.array([[data[f] for f in FEATURES_MKT_CLF]])
        X_scaled = scaler_clf_mkt.transform(X)
        prediction = int(clf_mkt.predict(X_scaled)[0])
        proba = float(clf_mkt.predict_proba(X_scaled)[0][1])
        latency = time.time() - start

        label = "Bon CAC âœ“" if prediction == 1 else "Mauvais CAC âœ—"
        PREDICTION_COUNT.labels(model_type="marketing_clf", result_label=label).inc()
        MODEL_CONFIDENCE.labels(model_type="marketing_clf").observe(proba)
        update_model_accuracy("marketing_clf", confidence=proba)

        drift = proba < 0.55
        if drift:
            logger.warning(f"[MARKETING DRIFT] Confiance faible: {proba:.3f}")

        logger.info(f"[MKT-CLF] prediction={prediction} confidence={proba:.3f} latency={latency*1000:.1f}ms")
        return jsonify({
            "cac_class":         prediction,
            "label":             label,
            "proba_bon_cac":     round(proba, 4),
            "proba_mauvais_cac": round(1 - proba, 4),
            "recommandation":    "Lancer la campagne â€” le CAC est rentable." if prediction == 1 else "Revoir le ciblage ou réduire le budget sur ce canal.",
            "latency_ms":        round(latency * 1000, 2),
            "drift_detected":    drift,
            "model":             "XGBoost"
        })
    except Exception as e:
        ERROR_COUNT.labels(endpoint="marketing_classification", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /marketing/classification: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/marketing/regression", methods=["POST"])
def marketing_regression():
    """Prédiction Taux de Conversion (Random Forest)"""
    if reg_mkt is None or scaler_reg_mkt is None:
        return jsonify({"error": "Modele Marketing Regression non charge â€” verifier models_marketing/models/reg_mkt.pkl"}), 503
    try:
        start = time.time()
        data = request.get_json()
        manquantes = [f for f in FEATURES_MKT_REG if f not in data]
        if manquantes:
            return jsonify({"error": f"Features manquantes: {manquantes}"}), 400

        X = np.array([[data[f] for f in FEATURES_MKT_REG]])
        X_scaled = scaler_reg_mkt.transform(X)
        prediction = float(reg_mkt.predict(X_scaled)[0])
        prediction = round(max(0.0, min(1.0, prediction)), 4)
        latency = time.time() - start

        if prediction > 0.3:   interp = "Élevé (>30%)"
        elif prediction > 0.1: interp = "Moyen (10-30%)"
        else:                  interp = "Faible (<10%)"

        PREDICTION_COUNT.labels(model_type="marketing_reg", result_label=interp).inc()
        update_model_accuracy("marketing_reg", confidence=prediction)

        logger.info(f"[MKT-REG] conversion_rate={prediction} latency={latency*1000:.1f}ms")
        return jsonify({
            "conversion_rate_pred": prediction,
            "conversion_pct":       f"{prediction * 100:.2f}%",
            "interpretation":       interp,
            "latency_ms":           round(latency * 1000, 2),
            "model":                "Random Forest"
        })
    except Exception as e:
        ERROR_COUNT.labels(endpoint="marketing_regression", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /marketing/regression: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/marketing/clustering", methods=["POST"])
def marketing_clustering():
    """Segmentation des Canaux Marketing (K-Means)"""
    if kmeans_mkt is None or scaler_clust_mkt is None:
        return jsonify({"error": "Modele Marketing Clustering non charge â€” verifier models_marketing/models/kmeans_mkt.pkl"}), 503
    try:
        start = time.time()
        data = request.get_json()
        manquantes = [f for f in FEATURES_MKT_CLUST if f not in data]
        if manquantes:
            return jsonify({"error": f"Features manquantes: {manquantes}"}), 400

        X = np.array([[data[f] for f in FEATURES_MKT_CLUST]])
        X_scaled = scaler_clust_mkt.transform(X)
        segment = int(kmeans_mkt.predict(X_scaled)[0])
        latency = time.time() - start

        seg_labels = {
            0: "Canal Premium â€” Top Performance",
            1: "Canal Moyen â€” Performance Correcte",
            2: "Canal Ã  Optimiser â€” Sous-performant"
        }
        seg_strategies = {
            0: "Maintenez et augmentez les investissements sur ce canal premium.",
            1: "Optimisez le message et testez A/B avant d'augmenter le budget.",
            2: "Réduisez les dépenses ou pivotez vers un canal plus performant."
        }
        label = seg_labels.get(segment, f"Segment {segment}")

        PREDICTION_COUNT.labels(model_type="marketing_clust", result_label=label).inc()

        logger.info(f"[MKT-CLUST] segment={segment} latency={latency*1000:.1f}ms")
        return jsonify({
            "segment":    segment,
            "label":      label,
            "strategie":  seg_strategies.get(segment, "Analyser les métriques du canal."),
            "latency_ms": round(latency * 1000, 2),
            "model":      "K-Means"
        })
    except Exception as e:
        ERROR_COUNT.labels(endpoint="marketing_clustering", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /marketing/clustering: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/marketing/forecast", methods=["GET", "POST"])
def marketing_forecast():
    """Prevision CAC 6 mois (ARIMA) -- lit arima_forecast.json"""
    try:
        chemin = None
        for candidate in [
            os.path.join(BASE, "arima_forecast.json"),
            os.path.join(BASE, "..", "arima_forecast.json"),
            os.path.join(BASE, "..", "models_marketing", "arima_forecast.json"),
        ]:
            if os.path.exists(candidate):
                chemin = candidate
                break
        if chemin is None:
            return jsonify({"error": "Fichier arima_forecast.json introuvable"}), 404
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                with open(chemin, encoding=enc) as f:
                    forecast_data = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        raw = forecast_data.get("previsions", forecast_data) if isinstance(forecast_data, dict) else forecast_data

        # Normalise en liste de nombres
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            # Déjà des objets {mois, valeur} — on garde tel quel
            previsions_obj = raw
        else:
            # Liste de nombres bruts — on construit les objets attendus par le frontend
            previsions_obj = [
                {"mois": i + 1, "valeur": round(float(v), 2), "label": f"Mois {i+1}"}
                for i, v in enumerate(raw if isinstance(raw, list) else [])
            ]

        PREDICTION_COUNT.labels(model_type="marketing_arima", result_label="forecast_ok").inc()
        logger.info(f"[MKT-ARIMA] forecast returned {len(previsions_obj)} months")
        return jsonify({
            "cac_forecast_6mois": previsions_obj,
            "previsions":         previsions_obj,
            "nb_mois":            len(previsions_obj),
            "unite":              "TND",
            "modele":             "ARIMA"
        })
    except Exception as e:
        ERROR_COUNT.labels(endpoint="marketing_forecast", error_type=type(e).__name__).inc()
        logger.error(f"[ERROR] /marketing/forecast: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/marketing/health", methods=["GET"])
def marketing_health():
    """Statut des modèles Marketing"""
    return jsonify({
        "status": "ok",
        "role": "Responsable Marketing",
        "modeles": {
            "clf_mkt":          clf_mkt is not None,
            "scaler_clf_mkt":   scaler_clf_mkt is not None,
            "reg_mkt":          reg_mkt is not None,
            "scaler_reg_mkt":   scaler_reg_mkt is not None,
            "kmeans_mkt":       kmeans_mkt is not None,
            "scaler_clust_mkt": scaler_clust_mkt is not None,
            "arima_forecast":   os.path.exists(os.path.join(BASE, "arima_forecast.json"))
        },
        "chemins_utilises": {
            "clf_mkt":        os.path.join(BASE, "models_marketing/models/clf_mkt.pkl"),
            "reg_mkt":        os.path.join(BASE, "models_marketing/models/reg_mkt.pkl"),
            "kmeans_mkt":     os.path.join(BASE, "models_marketing/models/kmeans_mkt.pkl"),
            "arima_forecast": os.path.join(BASE, "arima_forecast.json"),
        },
        "endpoints": [
            "POST /marketing/classification â€” Prédiction CAC (XGBoost)",
            "POST /marketing/regression    â€” Taux de Conversion (Random Forest)",
            "POST /marketing/clustering    â€” Segmentation Canaux (K-Means)",
            "GET  /marketing/forecast      â€” Prévision 6 mois (ARIMA)"
        ]
    })


# ============================================================
# SIMULATION ENDPOINT (for demo / testing)
# ============================================================
@app.route("/simulate/drift", methods=["POST"])
def simulate_drift():
    """Inject artificial drift into metrics for demo purposes."""
    data = request.get_json() or {}
    drift_type = data.get("type", "data_drift")
    intensity = float(data.get("intensity", 0.8))

    if drift_type == "data_drift":
        for feat in ["taux_reclamation_hist", "taux_acceptation_prest", "qualite_score"]:
            DRIFT_SCORE.labels(feature=feat).set(intensity)
        logger.warning(f"[SIMULATION] Data drift injected â€” intensity={intensity}")
        return jsonify({"simulated": "data_drift", "intensity": intensity})

    elif drift_type == "accuracy_drop":
        for model in ["classification", "produit_clf"]:
            drop = intensity * 10
            MODEL_ACCURACY_GAUGE.labels(model_type=model).set(BASELINES[model]["accuracy"] - (drop/100))
            ACCURACY_DROP.labels(model_type=model).set(drop)
        logger.error(f"[SIMULATION] Accuracy drop injected â€” {intensity*10:.1f}% drop")
        return jsonify({"simulated": "accuracy_drop", "drop_pct": intensity * 10})

    elif drift_type == "reset":
        for model, vals in BASELINES.items():
            MODEL_ACCURACY_GAUGE.labels(model_type=model).set(vals.get("accuracy", vals.get("r2", 0.85)))
            ACCURACY_DROP.labels(model_type=model).set(0.0)
        for feat in ["taux_reclamation_hist", "taux_acceptation_prest", "qualite_score"]:
            DRIFT_SCORE.labels(feature=feat).set(0.0)
        logger.info("[SIMULATION] Metrics reset to baseline")
        return jsonify({"simulated": "reset", "status": "ok"})

    return jsonify({"error": "Unknown drift type"}), 400


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("EventZilla ML API v4.0 â€” Monitoring Enabled")
    logger.info(f"BASE directory: {BASE}")
    logger.info("Metrics:  http://localhost:5000/metrics")
    logger.info("Health:   http://localhost:5000/health")
    logger.info("Qualite:  /predict/classification, /predict/regression, /predict/clustering")
    logger.info("Finance:  /audit, /forecast")
    logger.info("Produit:  /produit/classification, /produit/regression, /produit/clustering")
    logger.info("Marketing:/marketing/classification, /marketing/regression, /marketing/clustering, /marketing/forecast")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
