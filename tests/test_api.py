"""
tests/test_api.py
-----------------
Integration tests for the FastAPI scoring endpoints.

Uses FastAPI's TestClient — no running server required.
All endpoints and response fields aligned with src/api/main.py v1.0.0.

Endpoints tested:
  GET  /health
  GET  /model/info
  POST /predict
  POST /predict/batch
  GET  /customers/top-risk
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient with proper lifespan — loads model on startup."""
    with TestClient(app) as c:
        yield c


# ── Test customers ────────────────────────────────────────────────────────────

HIGH_RISK_CUSTOMER = {
    "type":              "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method":    "Electronic check",
    "gender":            "Female",
    "senior_citizen":    0,
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

LOW_RISK_CUSTOMER = {
    "type":              "Two year",
    "paperless_billing": "No",
    "payment_method":    "Bank transfer (automatic)",
    "gender":            "Male",
    "senior_citizen":    0,
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


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_threshold_in_range(self, client):
        resp = client.get("/health")
        threshold = resp.json()["threshold"]
        assert 0.0 < threshold < 1.0

    def test_health_has_model_field(self, client):
        resp = client.get("/health")
        assert "model" in resp.json()

    def test_health_has_features_field(self, client):
        resp = client.get("/health")
        assert "features" in resp.json()


# ── GET /model/info ───────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_has_auc(self, client):
        resp = client.get("/model/info")
        data = resp.json()
        assert "test_auc" in data
        assert 0.8 <= data["test_auc"] <= 1.0

    def test_model_info_has_threshold(self, client):
        resp = client.get("/model/info")
        data = resp.json()
        assert "theta_optimal" in data
        assert 0.0 < data["theta_optimal"] < 1.0

    def test_model_info_has_best_params(self, client):
        resp = client.get("/model/info")
        assert "best_params" in resp.json()


# ── POST /predict ─────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert resp.status_code == 200

    def test_predict_has_required_fields(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        data = resp.json()
        for field in ["p_churn", "flagged", "risk_tier", "threshold", "latency_ms"]:
            assert field in data, f"Missing field: {field}"

    def test_predict_probability_in_unit_interval(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        prob = resp.json()["p_churn"]
        assert 0.0 <= prob <= 1.0

    def test_predict_flagged_is_bool(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert isinstance(resp.json()["flagged"], bool)

    def test_predict_risk_tier_valid(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert resp.json()["risk_tier"] in ("High", "Medium", "Low")

    def test_predict_high_risk_scores_higher_than_low_risk(self, client):
        high = client.post("/predict", json=HIGH_RISK_CUSTOMER).json()["p_churn"]
        low  = client.post("/predict", json=LOW_RISK_CUSTOMER).json()["p_churn"]
        assert high > low

    def test_predict_flagged_consistent_with_probability(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        data = resp.json()
        expected = data["p_churn"] >= data["threshold"]
        assert data["flagged"] == expected

    def test_predict_missing_field_returns_422(self, client):
        bad = {k: v for k, v in HIGH_RISK_CUSTOMER.items() if k != "type"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_negative_monthly_charges_returns_422(self, client):
        bad = {**HIGH_RISK_CUSTOMER, "monthly_charges": -10.0}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_invalid_contract_type_returns_422(self, client):
        bad = {**HIGH_RISK_CUSTOMER, "type": "Weekly"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_with_customer_id_query_param(self, client):
        resp = client.post("/predict?customer_id=TEST-001", json=HIGH_RISK_CUSTOMER)
        assert resp.status_code == 200
        assert resp.json()["customer_id"] == "TEST-001"

    def test_predict_with_shap_returns_top_drivers(self, client):
        resp = client.post("/predict?include_shap=true", json=HIGH_RISK_CUSTOMER)
        assert resp.status_code == 200
        drivers = resp.json()["top_drivers"]
        assert isinstance(drivers, list)
        assert len(drivers) > 0
        assert "feature" in drivers[0]
        assert "shap_value" in drivers[0]
        assert "direction" in drivers[0]

    def test_predict_without_shap_returns_empty_drivers(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert resp.json()["top_drivers"] == []

    def test_predict_latency_ms_is_positive(self, client):
        resp = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert resp.json()["latency_ms"] > 0


# ── POST /predict/batch ───────────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_returns_200(self, client):
        payload = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_count_matches_input(self, client):
        payload = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
        resp = client.post("/predict/batch", json=payload)
        assert resp.json()["count"] == 2

    def test_batch_flagged_count_is_int(self, client):
        payload = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
        resp = client.post("/predict/batch", json=payload)
        assert isinstance(resp.json()["flagged"], int)

    def test_batch_results_sorted_descending(self, client):
        payload = [LOW_RISK_CUSTOMER, HIGH_RISK_CUSTOMER]  # low first in input
        resp = client.post("/predict/batch", json=payload)
        probs = [r["p_churn"] for r in resp.json()["results"]]
        assert probs == sorted(probs, reverse=True)

    def test_batch_all_probabilities_valid(self, client):
        payload = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER, HIGH_RISK_CUSTOMER]
        resp = client.post("/predict/batch", json=payload)
        for r in resp.json()["results"]:
            assert 0.0 <= r["p_churn"] <= 1.0

    def test_batch_all_risk_tiers_valid(self, client):
        payload = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
        resp = client.post("/predict/batch", json=payload)
        for r in resp.json()["results"]:
            assert r["risk_tier"] in ("High", "Medium", "Low")

    def test_batch_empty_list_returns_422(self, client):
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 422

    def test_batch_over_500_returns_422(self, client):
        payload = [HIGH_RISK_CUSTOMER] * 501
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 422


# ── GET /customers/top-risk ───────────────────────────────────────────────────

class TestTopRisk:
    def test_top_risk_returns_200(self, client):
        resp = client.get("/customers/top-risk")
        assert resp.status_code == 200

    def test_top_risk_default_count_is_20(self, client):
        resp = client.get("/customers/top-risk")
        assert resp.json()["count"] == 20

    def test_top_risk_custom_n(self, client):
        resp = client.get("/customers/top-risk?n=5")
        assert resp.json()["count"] == 5

    def test_top_risk_filter_by_high_tier(self, client):
        resp = client.get("/customers/top-risk?tier=High")
        assert resp.status_code == 200
        for c in resp.json()["customers"]:
            assert c["risk_tier"] == "High"

    def test_top_risk_invalid_tier_returns_422(self, client):
        resp = client.get("/customers/top-risk?tier=Critical")
        assert resp.status_code == 422

    def test_top_risk_has_required_customer_fields(self, client):
        resp = client.get("/customers/top-risk")
        customers = resp.json()["customers"]
        assert len(customers) > 0
        for field in ["customer_id", "p_churn", "flagged", "risk_tier"]:
            assert field in customers[0], f"Missing field: {field}"
