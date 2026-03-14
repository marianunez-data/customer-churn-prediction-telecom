"""
Churn Intelligence System — FastAPI
=====================================
Endpoints:
  GET  /health              — health check + model status
  GET  /model/info          — champion metadata (AUC, θ, params)
  POST /predict             — single customer prediction + SHAP top-5
  POST /predict/batch       — batch predictions ranked by risk
  GET  /customers/top-risk  — top-N riskiest from precomputed test set
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
# src/api/main.py → project root is 2 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# ── Load champion once at startup ─────────────────────────────────────────
def load_champion() -> dict:
    path = MODELS_DIR / "champion_calibrated.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    artifact = joblib.load(path)
    logger.info("Champion model loaded — θ=%.2f", artifact["deployed_threshold"])
    return artifact


def load_metadata() -> dict:
    path = REPORTS_DIR / "champion_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_test_predictions() -> pd.DataFrame:
    path = REPORTS_DIR / "final_results_test.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# Globals — loaded once
ARTIFACT    = load_champion()
METADATA    = load_metadata()
TEST_PREDS  = load_test_predictions()

PIPE      = ARTIFACT["base_pipeline"]
PLATT     = ARTIFACT["calibrator"]
THRESHOLD = ARTIFACT["deployed_threshold"]
CAT_COLS  = ARTIFACT["cat_cols"]
NUM_COLS  = ARTIFACT["num_cols"]
REQUIRED  = CAT_COLS + NUM_COLS

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Intelligence API",
    description=(
        "ML-powered customer churn prediction for Interconnect Telecom. "
        "LightGBM champion model with Platt calibration. AUC=0.9078, θ_opt=0.41."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """18 features required by the champion model."""
    # Categorical
    type:               str = Field(..., json_schema_extra={"example": "Month-to-month"})
    paperless_billing:  str = Field(..., json_schema_extra={"example": "Yes"})
    payment_method:     str = Field(..., json_schema_extra={"example": "Electronic check"})
    gender:             str = Field(..., json_schema_extra={"example": "Male"})
    partner:            str = Field(..., json_schema_extra={"example": "No"})
    dependents:         str = Field(..., json_schema_extra={"example": "No"})
    internet_service:   str = Field(..., json_schema_extra={"example": "Fiber optic"})
    online_security:    str = Field(..., json_schema_extra={"example": "No"})
    online_backup:      str = Field(..., json_schema_extra={"example": "No"})
    device_protection:  str = Field(..., json_schema_extra={"example": "No"})
    tech_support:       str = Field(..., json_schema_extra={"example": "No"})
    streaming_tv:       str = Field(..., json_schema_extra={"example": "No"})
    streaming_movies:   str = Field(..., json_schema_extra={"example": "No"})
    multiple_lines:     str = Field(..., json_schema_extra={"example": "No"})
    # Numeric
    senior_citizen:     int   = Field(..., ge=0, le=1)
    monthly_charges:    float = Field(..., gt=0)
    total_charges:      float = Field(..., ge=0)
    tenure_days:        int   = Field(..., ge=0)

    @field_validator("type")
    @classmethod
    def validate_contract(cls, v):
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"type must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "gender": "Male",
                "partner": "No",
                "dependents": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "multiple_lines": "No",
                "senior_citizen": 0,
                "monthly_charges": 70.35,
                "total_charges": 420.0,
                "tenure_days": 180,
            }
        }
    }


class BatchCustomer(CustomerFeatures):
    customer_id: Optional[str] = Field(None)


class PredictionResponse(BaseModel):
    customer_id:  Optional[str]
    p_churn:      float = Field(..., description="Calibrated churn probability [0,1]")
    flagged:      bool  = Field(..., description=f"True if p_churn >= θ_opt")
    risk_tier:    str   = Field(..., description="High / Medium / Low")
    threshold:    float
    top_drivers:  List[Dict[str, Any]] = Field(
        default=[], description="Top 5 SHAP feature contributions"
    )
    latency_ms:   float


class BatchPredictionResponse(BaseModel):
    count:        int
    flagged:      int
    results:      List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    algorithm:    str
    tuning:       str
    test_auc:     float
    val_auc:      float
    cv_auc_mean:  float
    cv_auc_std:   float
    test_brier:   float
    theta_optimal: float
    precision_at_theta: float
    recall_at_theta:    float
    train_rows:   int
    val_rows:     int
    test_rows:    int
    features:     int
    calibration:  str
    vc:           float
    cr:           float
    roi_optimal_test: float
    best_params:  Dict[str, Any]


# ── Helper functions ──────────────────────────────────────────────────────
def features_to_df(feat: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model → DataFrame with correct dtypes."""
    row = feat.model_dump()
    df  = pd.DataFrame([row])
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df[REQUIRED]


def run_inference(df: pd.DataFrame) -> np.ndarray:
    """Pipeline → Platt calibration → calibrated probabilities."""
    raw = PIPE.predict_proba(df)[:, 1]
    cal = PLATT.predict_proba(raw.reshape(-1, 1))[:, 1]
    return cal


def get_risk_tier(p: float) -> str:
    if p >= 0.60: return "High"
    if p >= 0.30: return "Medium"
    return "Low"


def compute_shap_top5(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute SHAP values and return top-5 drivers for one row."""
    try:
        import shap
        preprocessor = PIPE.named_steps["preprocessor"]
        classifier   = PIPE.named_steps["classifier"]
        X_t = preprocessor.transform(df)

        # Load feature names from saved file
        feat_path = REPORTS_DIR / "shap_feature_names.json"
        feat_names = (json.loads(feat_path.read_text())
                      if feat_path.exists()
                      else [f"f{i}" for i in range(X_t.shape[1])])

        explainer = shap.TreeExplainer(
            classifier, feature_perturbation="tree_path_dependent"
        )
        sv = explainer(X_t)
        vals = sv.values[0]

        # Top 5 by absolute value
        idx = np.argsort(np.abs(vals))[::-1][:5]
        return [
            {
                "feature":    feat_names[i],
                "shap_value": round(float(vals[i]), 4),
                "direction":  "increases_churn" if vals[i] > 0 else "reduces_churn",
            }
            for i in idx
        ]
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return []


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Health check — confirms model is loaded and ready."""
    return {
        "status":    "ok",
        "model":     "champion_calibrated.pkl",
        "threshold": THRESHOLD,
        "features":  len(REQUIRED),
        "api":       "1.0.0",
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Full champion model metadata — AUC, threshold, hyperparameters, ROI."""
    if not METADATA:
        raise HTTPException(status_code=404, detail="Metadata file not found.")
    return ModelInfoResponse(
        algorithm           = "LGBMClassifier",
        tuning              = "FLAML AutoML 180s",
        test_auc            = METADATA.get("test_auc",            0.9078),
        val_auc             = METADATA.get("val_auc",             0.9161),
        cv_auc_mean         = METADATA.get("cv_auc_mean",         0.9075),
        cv_auc_std          = METADATA.get("cv_auc_std",          0.0057),
        test_brier          = METADATA.get("test_brier",          0.1013),
        theta_optimal       = METADATA.get("theta_optimal",       0.41),
        precision_at_theta  = METADATA.get("precision_at_theta",  0.760),
        recall_at_theta     = METADATA.get("recall_at_theta",     0.695),
        train_rows          = METADATA.get("train_rows",          5282),
        val_rows            = METADATA.get("val_rows",            1056),
        test_rows           = METADATA.get("test_rows",           705),
        features            = len(REQUIRED),
        calibration         = "Platt Scaling (X_val)",
        vc                  = METADATA.get("Vc",                  69.0),
        cr                  = METADATA.get("Cr",                  20.0),
        roi_optimal_test    = METADATA.get("roi_optimal_test",    1617.0),
        best_params         = METADATA.get("best_params",         {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(
    customer: CustomerFeatures,
    customer_id: Optional[str] = Query(None, description="Optional customer ID"),
    include_shap: bool = Query(False, description="Include SHAP top-5 drivers"),
):
    """
    Score a single customer.

    Returns calibrated churn probability, risk tier, flag decision,
    and optionally the top-5 SHAP feature contributions.
    """
    t0 = time.perf_counter()

    try:
        df  = features_to_df(customer)
        cal = run_inference(df)[0]
    except Exception as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(status_code=422, detail=f"Inference failed: {e}")

    drivers = compute_shap_top5(df) if include_shap else []

    latency = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("predict | id=%s p=%.3f tier=%s latency=%.1fms",
                customer_id, cal, get_risk_tier(cal), latency)

    return PredictionResponse(
        customer_id = customer_id,
        p_churn     = round(float(cal), 4),
        flagged     = bool(cal >= THRESHOLD),
        risk_tier   = get_risk_tier(cal),
        threshold   = THRESHOLD,
        top_drivers = drivers,
        latency_ms  = latency,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(
    customers: List[BatchCustomer],
    include_shap: bool = Query(False, description="Include SHAP for each customer"),
):
    """
    Score a batch of customers (max 500).

    Returns results ranked by churn probability descending —
    ready to feed directly into a call-center priority queue.
    """
    if len(customers) == 0:
        raise HTTPException(status_code=422, detail="Empty batch.")
    if len(customers) > 500:
        raise HTTPException(status_code=422,
                            detail="Batch limit is 500 customers per request.")

    t0 = time.perf_counter()

    # Build DataFrame
    rows = []
    ids  = []
    for c in customers:
        d = c.model_dump()
        ids.append(d.pop("customer_id", None))
        rows.append(d)

    df_batch = pd.DataFrame(rows)
    for col in CAT_COLS:
        if col in df_batch.columns:
            df_batch[col] = df_batch[col].astype(str)
    for col in NUM_COLS:
        if col in df_batch.columns:
            df_batch[col] = pd.to_numeric(df_batch[col], errors="coerce").astype(float)
    df_batch = df_batch[REQUIRED]

    try:
        cals = run_inference(df_batch)
    except Exception as e:
        logger.error("Batch inference error: %s", e)
        raise HTTPException(status_code=422, detail=f"Batch inference failed: {e}")

    results = []
    for i, (cal, cid) in enumerate(zip(cals, ids)):
        drivers = compute_shap_top5(df_batch.iloc[[i]]) if include_shap else []
        results.append(PredictionResponse(
            customer_id = cid,
            p_churn     = round(float(cal), 4),
            flagged     = bool(cal >= THRESHOLD),
            risk_tier   = get_risk_tier(cal),
            threshold   = THRESHOLD,
            top_drivers = drivers,
            latency_ms  = 0.0,
        ))

    # Sort by risk descending
    results.sort(key=lambda r: r.p_churn, reverse=True)

    latency = round((time.perf_counter() - t0) * 1000, 2)
    flagged_count = sum(1 for r in results if r.flagged)
    logger.info("batch | n=%d flagged=%d latency=%.1fms",
                len(results), flagged_count, latency)

    return BatchPredictionResponse(
        count   = len(results),
        flagged = flagged_count,
        results = results,
    )


@app.get("/customers/top-risk", tags=["Analytics"])
def top_risk_customers(
    n:    int = Query(20,    ge=1, le=705, description="Number of customers to return"),
    tier: Optional[str] = Query(None, description="Filter by risk tier: High/Medium/Low"),
):
    """
    Return top-N riskiest customers from the precomputed test set.

    Useful for daily call-center briefing — no re-scoring needed.
    """
    if TEST_PREDS.empty:
        raise HTTPException(status_code=404,
                            detail="Precomputed predictions not found.")

    df = TEST_PREDS.copy()

    if tier:
        tier_clean = tier.capitalize()
        if tier_clean not in {"High", "Medium", "Low"}:
            raise HTTPException(status_code=422,
                                detail="tier must be High, Medium, or Low")
        df = df[df["risk_tier"] == tier_clean]

    df = df.sort_values("y_cal", ascending=False).head(n)

    return {
        "count":   len(df),
        "filter":  tier or "all",
        "customers": [
            {
                "customer_id": row["customer_id"],
                "p_churn":     round(row["y_cal"], 4),
                "flagged":     bool(row["y_pred_optimal"] == 1),
                "risk_tier":   row["risk_tier"],
                "true_label":  int(row["y_true"]) if "y_true" in row else None,
            }
            for _, row in df.iterrows()
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
