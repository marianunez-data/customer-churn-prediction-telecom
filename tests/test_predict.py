"""
tests/test_predict.py
---------------------
Unit tests for src/models/predict.py — inference utilities.

load_champion() returns a 4-tuple:
    (base_pipeline, calibrator, deployed_threshold, metadata_dict)

Threshold is read from:
    1. champion_calibrated.pkl  → deployed_threshold key
    2. threshold_report.json    → theta_optimal key
    3. champion_metadata.json   → deployed_threshold key
    4. fallback                 → 0.50
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def champion_assets():
    """
    Load champion artifact dict.
    Keys: base_pipeline, calibrator, deployed_threshold, cat_cols, num_cols, ...
    """
    from src.models.predict import load_champion
    return load_champion()


@pytest.fixture(scope="module")
def high_risk_customer() -> dict:
    """High-risk customer — month-to-month, fiber optic, short tenure."""
    return {
        "type":              "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method":    "Electronic check",
        "gender":            "Female",
        "senior_citizen":    0,          # int — as pipeline expects
        "partner":           "Yes",
        "dependents":        "No",
        "internet_service":  "Fiber optic",
        "online_security":   "No",
        "online_backup":     "No",
        "device_protection": "No",
        "tech_support":      "No",
        "streaming_tv":      "Yes",
        "streaming_movies":  "Yes",
        "multiple_lines":    "No",
        "monthly_charges":   85.5,
        "total_charges":     2100.0,
        "tenure_days":       180,
    }


@pytest.fixture(scope="module")
def low_risk_customer() -> dict:
    """Low-risk customer — two-year contract, DSL, long tenure."""
    return {
        "type":              "Two year",
        "paperless_billing": "No",
        "payment_method":    "Bank transfer (automatic)",
        "gender":            "Male",
        "senior_citizen":    0,          # int — as pipeline expects
        "partner":           "Yes",
        "dependents":        "Yes",
        "internet_service":  "DSL",
        "online_security":   "Yes",
        "online_backup":     "Yes",
        "device_protection": "Yes",
        "tech_support":      "Yes",
        "streaming_tv":      "No",
        "streaming_movies":  "No",
        "multiple_lines":    "No",
        "monthly_charges":   35.0,
        "total_charges":     8500.0,
        "tenure_days":       2000,
    }


# ── Tests: load_champion ──────────────────────────────────────────────────────

class TestLoadChampion:
    def test_returns_dict(self, champion_assets):
        assert isinstance(champion_assets, dict)

    def test_pipeline_has_predict_proba(self, champion_assets):
        pipeline = champion_assets["base_pipeline"]
        assert hasattr(pipeline, "predict_proba")

    def test_calibrator_has_predict_proba(self, champion_assets):
        calibrator = champion_assets.get("calibrator")
        if calibrator is not None:
            assert hasattr(calibrator, "predict_proba")

    def test_threshold_is_float_in_range(self, champion_assets):
        threshold = champion_assets["deployed_threshold"]
        assert isinstance(threshold, float)
        assert 0.0 < threshold < 1.0

    def test_threshold_matches_deployed_value(self, champion_assets):
        """Deployed threshold should be 0.41 — the OOF-optimized value."""
        threshold = champion_assets["deployed_threshold"]
        assert abs(threshold - 0.41) < 0.05, \
            f"Expected threshold near 0.41, got {threshold}"

    def test_artifact_has_cat_cols(self, champion_assets):
        assert "cat_cols" in champion_assets

    def test_artifact_has_num_cols(self, champion_assets):
        assert "num_cols" in champion_assets

    def test_pipeline_has_preprocessor_step(self, champion_assets):
        pipeline = champion_assets["base_pipeline"]
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_has_classifier_step(self, champion_assets):
        pipeline = champion_assets["base_pipeline"]
        assert "classifier" in pipeline.named_steps


# ── Tests: get_risk_tier ──────────────────────────────────────────────────────

class TestGetRiskTier:
    def test_high_tier(self):
        from src.models.predict import get_risk_tier
        assert get_risk_tier(0.75, 0.41) == "High"

    def test_high_tier_at_boundary(self):
        from src.models.predict import get_risk_tier
        assert get_risk_tier(0.60, 0.41) == "High"

    def test_medium_tier(self):
        from src.models.predict import get_risk_tier
        assert get_risk_tier(0.45, 0.41) == "Medium"

    def test_medium_tier_at_boundary(self):
        from src.models.predict import get_risk_tier
        assert get_risk_tier(0.30, 0.41) == "Medium"

    def test_low_tier(self):
        from src.models.predict import get_risk_tier
        assert get_risk_tier(0.10, 0.41) == "Low"

    def test_boundary_zero_is_low(self):
        from src.models.predict import get_risk_tier
        assert get_risk_tier(0.0, 0.41) == "Low"

    def test_boundary_one_is_high(self):
        from src.models.predict import get_risk_tier
        assert get_risk_tier(1.0, 0.41) == "High"


# ── Tests: score_customer ─────────────────────────────────────────────────────

class TestScoreCustomer:
    def test_returns_required_keys(self, champion_assets, high_risk_customer):
        from src.models.predict import score_customer
        result = score_customer(champion_assets, high_risk_customer)
        for key in ["churn_probability", "churn_flag", "risk_tier",
                    "threshold_used", "calibrated"]:
            assert key in result, f"Missing key: {key}"

    def test_probability_in_unit_interval(self, champion_assets, high_risk_customer):
        from src.models.predict import score_customer
        result = score_customer(champion_assets, high_risk_customer)
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_flag_is_binary(self, champion_assets, high_risk_customer):
        from src.models.predict import score_customer
        result = score_customer(champion_assets, high_risk_customer)
        assert result["churn_flag"] in (0, 1)

    def test_risk_tier_is_valid(self, champion_assets, high_risk_customer):
        from src.models.predict import score_customer
        result = score_customer(champion_assets, high_risk_customer)
        assert result["risk_tier"] in ("High", "Medium", "Low")

    def test_flag_consistent_with_probability(self, champion_assets, high_risk_customer):
        from src.models.predict import score_customer
        result = score_customer(champion_assets, high_risk_customer)
        threshold = champion_assets["deployed_threshold"]
        expected = int(result["churn_probability"] >= threshold)
        assert result["churn_flag"] == expected

    def test_threshold_used_matches_deployed(self, champion_assets, high_risk_customer):
        from src.models.predict import score_customer
        result = score_customer(champion_assets, high_risk_customer)
        assert result["threshold_used"] == champion_assets["deployed_threshold"]

    def test_high_risk_scores_higher_than_low_risk(
        self, champion_assets, high_risk_customer, low_risk_customer
    ):
        from src.models.predict import score_customer
        high = score_customer(champion_assets, high_risk_customer)
        low  = score_customer(champion_assets, low_risk_customer)
        assert high["churn_probability"] > low["churn_probability"], \
            "High-risk customer must score higher than low-risk customer"

    def test_long_tenure_lowers_churn_score(self, champion_assets, high_risk_customer):
        from src.models.predict import score_customer
        long_tenure = {**high_risk_customer, "tenure_days": 2500,
                       "total_charges": 15000.0}
        short = score_customer(champion_assets, high_risk_customer)
        long  = score_customer(champion_assets, long_tenure)
        assert short["churn_probability"] > long["churn_probability"], \
            "Short tenure must produce higher churn probability than long tenure"

    def test_calibrated_flag_is_true_when_calibrator_loaded(
        self, champion_assets, high_risk_customer
    ):
        from src.models.predict import score_customer
        if champion_assets.get("calibrator") is not None:
            result = score_customer(champion_assets, high_risk_customer)
            assert result["calibrated"] is True


# ── Tests: score_batch ────────────────────────────────────────────────────────

class TestScoreBatch:
    def test_returns_dataframe(self, champion_assets, high_risk_customer, low_risk_customer):
        from src.models.predict import score_batch
        df = pd.DataFrame([high_risk_customer, low_risk_customer])
        result = score_batch(champion_assets, df)
        assert isinstance(result, pd.DataFrame)

    def test_output_row_count_matches_input(self, champion_assets, high_risk_customer):
        from src.models.predict import score_batch
        n = 5
        df = pd.DataFrame([high_risk_customer] * n)
        result = score_batch(champion_assets, df)
        assert len(result) == n

    def test_output_has_required_columns(self, champion_assets, high_risk_customer):
        from src.models.predict import score_batch
        df = pd.DataFrame([high_risk_customer])
        result = score_batch(champion_assets, df)
        for col in ["churn_probability", "churn_flag", "risk_tier"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_descending_by_probability(
        self, champion_assets, high_risk_customer, low_risk_customer
    ):
        from src.models.predict import score_batch
        df = pd.DataFrame([low_risk_customer, high_risk_customer])
        result = score_batch(champion_assets, df)
        probs = result["churn_probability"].tolist()
        assert probs == sorted(probs, reverse=True), \
            "Batch results must be sorted descending by probability"

    def test_all_probabilities_in_unit_interval(
        self, champion_assets, high_risk_customer, low_risk_customer
    ):
        from src.models.predict import score_batch
        df = pd.DataFrame([high_risk_customer, low_risk_customer])
        result = score_batch(champion_assets, df)
        assert (result["churn_probability"] >= 0.0).all()
        assert (result["churn_probability"] <= 1.0).all()

    def test_customer_id_preserved_when_present(
        self, champion_assets, high_risk_customer
    ):
        from src.models.predict import score_batch
        df = pd.DataFrame([{**high_risk_customer, "customer_id": "TEST-001"}])
        result = score_batch(champion_assets, df)
        assert "customer_id" in result.columns
        assert "TEST-001" in result["customer_id"].values

    def test_all_risk_tiers_valid(
        self, champion_assets, high_risk_customer, low_risk_customer
    ):
        from src.models.predict import score_batch
        df = pd.DataFrame([high_risk_customer, low_risk_customer])
        result = score_batch(champion_assets, df)
        assert result["risk_tier"].isin(["High", "Medium", "Low"]).all()


# ── Tests: threshold_report integrity ────────────────────────────────────────

class TestThresholdReport:
    def test_threshold_report_exists(self):
        path = PROJECT_ROOT / "reports" / "threshold_report.json"
        assert path.exists(), \
            "threshold_report.json not found — run notebook §6.1 first"

    def test_threshold_report_has_theta_optimal(self):
        path = PROJECT_ROOT / "reports" / "threshold_report.json"
        data = json.loads(path.read_text())
        assert "theta_optimal" in data, \
            "threshold_report.json missing 'theta_optimal' key"

    def test_deployed_threshold_is_valid(self):
        path = PROJECT_ROOT / "reports" / "threshold_report.json"
        data = json.loads(path.read_text())
        t = data.get("theta_optimal", data.get("deployed_threshold", None))
        assert t is not None, "No threshold key found in threshold_report.json"
        assert 0.0 < float(t) < 1.0

    def test_threshold_report_has_business_params(self):
        path = PROJECT_ROOT / "reports" / "threshold_report.json"
        data = json.loads(path.read_text())
        # business params may be at root level or nested under 'business_params'
        params = data.get("business_params", data)
        for key in ["Vc", "Cr"]:
            assert key in params, \
                f"Missing business param '{key}' in threshold_report.json"


# ── Tests: champion_metadata integrity ───────────────────────────────────────

class TestChampionMetadata:
    def test_metadata_exists(self):
        path = PROJECT_ROOT / "reports" / "champion_metadata.json"
        assert path.exists(), "champion_metadata.json not found"

    def test_metadata_has_auc(self):
        path = PROJECT_ROOT / "reports" / "champion_metadata.json"
        data = json.loads(path.read_text())
        assert "val_auc" in data or "test_auc" in data, \
            "champion_metadata.json missing AUC metrics"

    def test_test_auc_above_baseline(self):
        path = PROJECT_ROOT / "reports" / "champion_metadata.json"
        data = json.loads(path.read_text())
        auc = data.get("test_auc", data.get("val_auc", 0))
        assert float(auc) >= 0.85, \
            f"Test AUC {auc} is below minimum expected threshold of 0.85"
