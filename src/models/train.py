"""
src/models/train.py
-------------------
Standalone retraining script for the champion LightGBM churn model.

Usage:
    python -m src.models.train                   # train with FLAML AutoML
    python -m src.models.train --no-automl       # use stored best params
    python -m src.models.train --experiment dev  # custom MLflow experiment name

Outputs:
    models/champion_lgbm.pkl          – serialized sklearn Pipeline
    reports/champion_metadata.json    – metrics + hyperparameters
    mlruns/                           – MLflow experiment tracking (if available)
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_modeling_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load certified modeling parquet and return X, y."""
    df = pd.read_parquet(path)
    drop = [c for c in ["customer_id", "tenure_months", "tenure_range"] if c in df.columns]
    X = df.drop(columns=["churn"] + drop)
    y = df["churn"]
    return X, y


def split_data(X, y, val_size=0.15, test_size=0.10, random_state=42):
    """Stratified three-way split: train / val / test."""
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )


def evaluate(pipeline, X, y, split_name="val") -> dict:
    """Return dict of evaluation metrics on a given split."""
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    return {
        f"{split_name}_auc":       round(float(roc_auc_score(y, y_prob)), 4),
        f"{split_name}_pr_auc":    round(float(average_precision_score(y, y_prob)), 4),
        f"{split_name}_recall":    round(float(recall_score(y, y_pred)), 4),
        f"{split_name}_precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        f"{split_name}_f1":        round(float(f1_score(y, y_pred)), 4),
    }


# ── FLAML AutoML search ───────────────────────────────────────────────────────

def run_automl(X_train_enc, y_train, X_val_enc, y_val, budget_seconds=120):
    """Run FLAML search; return best LightGBM params dict."""
    try:
        from flaml import AutoML
    except ImportError:
        print("  FLAML not installed — using default LightGBM params.")
        return {}

    automl = AutoML()
    automl.fit(
        X_train_enc, y_train,
        X_val=X_val_enc, y_val=y_val,
        task="classification",
        metric="roc_auc",
        estimator_list=["lgbm"],
        time_budget=budget_seconds,
        verbose=0,
    )
    cfg = automl.best_config
    lgbm_keys = {
        "n_estimators", "num_leaves", "min_child_samples", "learning_rate",
        "log_max_bin", "colsample_bytree", "reg_alpha", "reg_lambda",
    }
    return {k: v for k, v in cfg.items() if k in lgbm_keys}


# ── Main training pipeline ────────────────────────────────────────────────────

def train(
    modeling_path: Path | None = None,
    model_out: Path | None = None,
    meta_out: Path | None = None,
    use_automl: bool = True,
    automl_budget: int = 120,
    experiment_name: str = "churn-prediction",
) -> dict:
    """
    Full training pipeline.

    Returns
    -------
    dict
        Champion metadata including metrics and hyperparameters.
    """
    modeling_path = modeling_path or PROJECT_ROOT / "data" / "processed" / "df_modeling.parquet"
    model_out     = model_out     or PROJECT_ROOT / "models" / "champion_lgbm.pkl"
    meta_out      = meta_out      or PROJECT_ROOT / "reports" / "champion_metadata.json"

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"  Loading data: {modeling_path.relative_to(PROJECT_ROOT)}")
    X, y = load_modeling_data(modeling_path)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    print(f"  Shape: {X.shape} | Churn rate: {y.mean():.1%}")
    print(f"  Features: {len(cat_cols)} categorical + {len(num_cols)} numeric")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"  Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

    # ── 3. Preprocessor ───────────────────────────────────────────────────────
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_enc  = preprocessor.fit_transform(X_train)
    X_val_enc    = preprocessor.transform(X_val)

    # ── 4. Hyperparameter search ──────────────────────────────────────────────
    best_params: dict = {}
    search_time = 0.0

    if use_automl:
        print(f"  Running FLAML AutoML ({automl_budget}s budget)...")
        t0 = time.time()
        best_params = run_automl(X_train_enc, y_train, X_val_enc, y_val, automl_budget)
        search_time = round(time.time() - t0, 1)
        print(f"  FLAML completed in {search_time}s | params: {best_params}")
    else:
        print("  Using stored best params (no AutoML search).")
        existing_meta = meta_out if meta_out.exists() else None
        if existing_meta:
            with open(existing_meta) as f:
                meta = json.load(f)
            best_params = meta.get("best_params", {})

    # ── 5. Train champion pipeline ────────────────────────────────────────────
    lgbm_params = {
        "n_estimators":   best_params.get("n_estimators", 500),
        "num_leaves":     best_params.get("num_leaves", 31),
        "learning_rate":  best_params.get("learning_rate", 0.05),
        "colsample_bytree": best_params.get("colsample_bytree", 0.8),
        "reg_alpha":      best_params.get("reg_alpha", 0.001),
        "reg_lambda":     best_params.get("reg_lambda", 0.01),
        "min_child_samples": best_params.get("min_child_samples", 20),
        "is_unbalance":   True,
        "random_state":   42,
        "n_jobs":         -1,
        "verbose":        -1,
    }

    champion = Pipeline([
        ("preprocessor", build_preprocessor(cat_cols, num_cols)),
        ("classifier",   LGBMClassifier(**lgbm_params)),
    ])

    t0 = time.time()
    champion.fit(X_train, y_train)
    fit_time = round(time.time() - t0, 2)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    val_metrics  = evaluate(champion, X_val,  y_val,  "val")
    test_metrics = evaluate(champion, X_test, y_test, "test")

    print(f"\n  ─── CHAMPION METRICS ───────────────────────────────")
    for k, v in val_metrics.items():
        print(f"  {k:<22}: {v}")
    print(f"  {'fit_time_sec':<22}: {fit_time}")
    print(f"  {'search_time_sec':<22}: {search_time}")

    # ── 7. Serialize ──────────────────────────────────────────────────────────
    model_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(champion, model_out)
    print(f"\n  Saved: {model_out.relative_to(PROJECT_ROOT)}")

    metadata = {
        "model_name":    "champion_lgbm",
        "algorithm":     "LightGBM",
        "tuning_method": "FLAML AutoML" if use_automl else "stored params",
        "imbalance_method": "is_unbalance=True",
        **val_metrics,
        **test_metrics,
        "best_params":   lgbm_params,
        "search_time_sec": search_time,
        "fit_time_sec":  fit_time,
        "train_rows":    len(X_train),
        "val_rows":      len(X_val),
        "test_rows":     len(X_test),
        "n_features":    len(cat_cols) + len(num_cols),
        "features":      cat_cols + num_cols,
        "cat_cols":      cat_cols,
        "num_cols":      num_cols,
        "trained_on":    pd.Timestamp.now().isoformat(),
        "model_path":    str(model_out.relative_to(PROJECT_ROOT)),
    }
    meta_out.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"  Saved: {meta_out.relative_to(PROJECT_ROOT)}")

    # ── 8. MLflow logging (optional) ──────────────────────────────────────────
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="champion_lgbm"):
            mlflow.log_params(lgbm_params)
            mlflow.log_metrics({k.replace("val_", ""): v for k, v in val_metrics.items()})
            mlflow.log_artifact(str(model_out))
            mlflow.log_artifact(str(meta_out))
        print(f"  MLflow run logged to experiment '{experiment_name}'")
    except Exception:
        pass  # MLflow optional — training succeeds without it

    return metadata


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train champion churn model")
    p.add_argument("--no-automl", action="store_true", help="Skip FLAML, use stored params")
    p.add_argument("--budget", type=int, default=120, help="FLAML budget in seconds")
    p.add_argument("--experiment", default="churn-prediction", help="MLflow experiment name")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("\n  INTERCONNECT CHURN MODEL — TRAINING PIPELINE")
    print(f"  {'─'*50}")
    train(
        use_automl=not args.no_automl,
        automl_budget=args.budget,
        experiment_name=args.experiment,
    )
    print("\n  Training complete.")
