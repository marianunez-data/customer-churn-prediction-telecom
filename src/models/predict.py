"""
src/models/predict.py
---------------------
Inference utilities for the Churn Intelligence System.

Public API:
    load_champion()                    → artifact dict
    get_risk_tier(p, theta)            → "High" | "Medium" | "Low"
    score_customer(artifact, row_dict) → dict with prediction results
    score_batch(artifact, df)          → DataFrame ranked by churn_probability
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"


def load_champion() -> dict:
    """
    Load champion_calibrated.pkl and return the artifact dict.

    Keys guaranteed:
        base_pipeline       — fitted sklearn Pipeline
        calibrator          — fitted CalibratedClassifierCV (Platt)
        deployed_threshold  — float, θ_optimal
        cat_cols            — list of categorical feature names
        num_cols            — list of numeric feature names
    """
    path = MODELS_DIR / "champion_calibrated.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"champion_calibrated.pkl not found at {path}. "
            "Run notebook §5 first."
        )
    artifact = joblib.load(path)
    logger.info(
        "Champion loaded — θ=%.2f | pipeline steps: %s",
        artifact.get("deployed_threshold", "?"),
        list(artifact.get("base_pipeline", {}).named_steps.keys()),
    )
    return artifact


def get_risk_tier(p: float, theta: float = 0.41) -> str:
    """
    Map calibrated probability to a business risk tier.

    Tiers (fixed business thresholds, independent of θ):
        High    p >= 0.60   — flagged, outreach this week
        Medium  p >= 0.30   — monitor, include in next campaign
        Low     p <  0.30   — no action needed
    """
    if p >= 0.60: return "High"
    if p >= 0.30: return "Medium"
    return "Low"


def _prepare_df(artifact: dict, row: dict) -> pd.DataFrame:
    cat_cols = artifact["cat_cols"]
    num_cols = artifact["num_cols"]
    df = pd.DataFrame([row])
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df[cat_cols + num_cols]


def _run_inference(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    raw = artifact["base_pipeline"].predict_proba(df)[:, 1]
    cal = artifact["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    return cal


def score_customer(artifact: dict, row: dict) -> Dict[str, Any]:
    """
    Score a single customer.

    Returns dict with:
        churn_probability, churn_flag, risk_tier, threshold_used, calibrated
    """
    theta = artifact["deployed_threshold"]
    df    = _prepare_df(artifact, row)
    cal   = _run_inference(artifact, df)[0]
    return {
        "churn_probability": round(float(cal), 4),
        "churn_flag":        int(cal >= theta),
        "risk_tier":         get_risk_tier(cal, theta),
        "threshold_used":    theta,
        "calibrated":        artifact.get("calibrator") is not None,
    }


def score_batch(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch of customers. Returns DataFrame sorted descending by
    churn_probability. Preserves customer_id column if present.
    """
    theta    = artifact["deployed_threshold"]
    cat_cols = artifact["cat_cols"]
    num_cols = artifact["num_cols"]
    required = cat_cols + num_cols

    has_id = "customer_id" in df.columns
    id_col = df["customer_id"].reset_index(drop=True) if has_id else None

    df_feat = df.copy()
    for c in cat_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].astype(str)
    for c in num_cols:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").astype(float)

    cals = _run_inference(artifact, df_feat[required])

    result = pd.DataFrame({
        "churn_probability": np.round(cals, 4),
        "churn_flag":        (cals >= theta).astype(int),
        "risk_tier":         [get_risk_tier(p, theta) for p in cals],
    })
    if has_id:
        result.insert(0, "customer_id", id_col)

    return result.sort_values("churn_probability", ascending=False).reset_index(drop=True)
