# tests/test_transform.py
"""
Unit tests for ChurnPreprocessor and engineer_features.

These tests validate:
  - Happy path: clean data flows correctly end to end
  - Edge cases: special chars, missing columns, wrong dtypes
  - Leakage prevention: dates never reach modeling DataFrame
  - Idempotency: running twice gives same result
"""
import pandas as pd
import pytest

from src.etl.ingest import load_raw_data
from src.etl.transform import SERVICE_COLS, ChurnPreprocessor
from src.features.engineer import (
    CUTOFF_DATE,
    OBS_START,
    classify_segment,
    compute_tenure,
    engineer_features,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df_raw():
    return load_raw_data("data/raw")


@pytest.fixture(scope="module")
def df_clean(df_raw):
    return ChurnPreprocessor().fit_transform(df_raw)


@pytest.fixture(scope="module")
def df_model(df_clean):
    return engineer_features(df_clean)


@pytest.fixture
def minimal_raw_row() -> pd.DataFrame:
    """Single valid customer record in raw PascalCase format."""
    return pd.DataFrame([{
        "customerID"      : "TEST-0001",
        "BeginDate"       : "2018-01-01",
        "EndDate"         : "No",
        "Type"            : "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod"   : "Electronic check",
        "MonthlyCharges"  : 65.0,
        "TotalCharges"    : "780.0",
        "gender"          : "Female",
        "SeniorCitizen"   : 0,
        "Partner"         : "Yes",
        "Dependents"      : "No",
        "InternetService" : "Fiber optic",
        "OnlineSecurity"  : "No",
        "OnlineBackup"    : "No",
        "DeviceProtection": "No",
        "TechSupport"     : "No",
        "StreamingTV"     : "Yes",
        "StreamingMovies" : "Yes",
        "MultipleLines"   : "No",
    }])


# ── ChurnPreprocessor: column standardization ──────────────────────────────────

class TestColumnStandardization:
    def test_columns_are_snake_case(self, df_clean):
        for col in df_clean.columns:
            assert col == col.lower(), \
                f"Column '{col}' is not snake_case."
            assert " " not in col, \
                f"Column '{col}' contains spaces."

    def test_customer_id_renamed(self, df_clean):
        assert "customer_id" in df_clean.columns
        assert "customerID"  not in df_clean.columns

    def test_column_count_preserved(self, df_raw, df_clean):
        assert df_clean.shape[1] == df_raw.shape[1]


# ── ChurnPreprocessor: type casting ───────────────────────────────────────────

class TestTypeCasting:
    def test_begin_date_is_datetime(self, df_clean):
        assert df_clean["begin_date"].dtype == "datetime64[ns]", \
            f"begin_date dtype: {df_clean['begin_date'].dtype}"

    def test_end_date_is_datetime(self, df_clean):
        assert df_clean["end_date"].dtype == "datetime64[ns]", \
            f"end_date dtype: {df_clean['end_date'].dtype}"

    def test_total_charges_is_float(self, df_clean):
        assert df_clean["total_charges"].dtype == "float64"

    def test_categoricals_have_category_dtype(self, df_clean):
        expected_cats = [
            "type", "gender", "senior_citizen", "internet_service",
            "contract" if "contract" in df_clean.columns else "type",
        ]
        for col in ["type", "gender", "senior_citizen"]:
            assert str(df_clean[col].dtype) == "category", \
                f"'{col}' should be category, got {df_clean[col].dtype}"


# ── ChurnPreprocessor: null imputation ────────────────────────────────────────

class TestNullImputation:
    def test_no_nulls_except_end_date(self, df_clean):
        cols = df_clean.columns.difference(["end_date"])
        null_counts = df_clean[cols].isnull().sum()
        assert null_counts.sum() == 0, \
            f"Unexpected nulls: {null_counts[null_counts > 0].to_dict()}"

    def test_service_nulls_filled(self, df_clean):
        for col in SERVICE_COLS:
            if col in df_clean.columns:
                assert df_clean[col].isnull().sum() == 0, \
                    f"Service column '{col}' still has nulls."

    def test_total_charges_no_nulls(self, df_clean):
        assert df_clean["total_charges"].isnull().sum() == 0

    def test_senior_citizen_is_yes_no(self, df_clean):
        unique_vals = set(df_clean["senior_citizen"].cat.categories)
        assert unique_vals == {"No", "Yes"}, \
            f"senior_citizen values: {unique_vals}"


# ── ChurnPreprocessor: special character detection ────────────────────────────

class TestSpecialCharDetection:
    def test_raises_on_special_chars(self, minimal_raw_row):
        bad = minimal_raw_row.copy()
        bad["Type"] = "Month-to-month@hack"
        with pytest.raises(ValueError, match="Special characters detected"):
            ChurnPreprocessor().fit_transform(bad)

    def test_clean_data_passes(self, minimal_raw_row):
        """No exception should be raised on clean data."""
        result = ChurnPreprocessor().fit_transform(minimal_raw_row)
        assert len(result) == 1


# ── ChurnPreprocessor: idempotency ────────────────────────────────────────────

class TestIdempotency:
    def test_shape_unchanged_on_second_run(self, df_raw):
        pre    = ChurnPreprocessor()
        first  = pre.fit_transform(df_raw)
        second = pre.fit_transform(df_raw)
        assert first.shape == second.shape

    def test_values_unchanged_on_second_run(self, df_raw):
        pre    = ChurnPreprocessor()
        first  = pre.fit_transform(df_raw)
        second = pre.fit_transform(df_raw)
        pd.testing.assert_frame_equal(first, second)


# ── engineer_features: target engineering ─────────────────────────────────────

class TestTargetEngineering:
    def test_churn_is_binary(self, df_model):
        assert df_model["churn"].isin([0, 1]).all()

    def test_churn_rate_in_expected_range(self, df_model):
        rate = df_model["churn"].mean()
        assert 0.20 <= rate <= 0.35, \
            f"Churn rate {rate:.1%} outside expected range [20%, 35%]."

    def test_active_customers_have_churn_zero(self, df_clean, df_model):
        active_mask = df_clean["end_date"].isna()
        assert (df_model.loc[active_mask, "churn"] == 0).all()

    def test_churned_customers_have_churn_one(self, df_clean, df_model):
        churned_mask = df_clean["end_date"].notna()
        assert (df_model.loc[churned_mask, "churn"] == 1).all()


# ── engineer_features: tenure ─────────────────────────────────────────────────

class TestTenureEngineering:
    def test_tenure_non_negative(self, df_model):
        assert (df_model["tenure_days"] >= 0).all()

    def test_tenure_max_reasonable(self, df_model):
        max_possible = (CUTOFF_DATE - pd.Timestamp("2010-01-01")).days
        assert df_model["tenure_days"].max() <= max_possible

    def test_compute_tenure_helper(self):
        begin = pd.Timestamp("2019-02-01")
        result = compute_tenure(begin, reference_date=CUTOFF_DATE)
        assert result == 365


# ── engineer_features: leakage prevention ────────────────────────────────────

class TestLeakagePrevention:
    def test_end_date_not_in_model(self, df_model):
        assert "end_date"   not in df_model.columns, \
            "LEAKAGE: end_date present in modeling DataFrame."

    def test_begin_date_not_in_model(self, df_model):
        assert "begin_date" not in df_model.columns, \
            "LEAKAGE: begin_date present in modeling DataFrame."


# ── engineer_features: segmentation ──────────────────────────────────────────

class TestSegmentation:
    def test_segment_values_valid(self, df_model):
        assert df_model["segment"].isin(["eligible", "new"]).all()

    def test_eligible_count(self, df_model):
        assert (df_model["segment"] == "eligible").sum() == 6105

    def test_new_count(self, df_model):
        assert (df_model["segment"] == "new").sum() == 938

    @pytest.mark.parametrize("begin,expected", [
        (pd.Timestamp("2018-01-01"), "eligible"),
        (pd.Timestamp("2019-08-31"), "eligible"),
        (pd.Timestamp("2019-10-01"), "new"),
        (pd.Timestamp("2020-01-15"), "new"),
    ])
    def test_classify_segment_boundaries(self, begin, expected):
        assert classify_segment(begin) == expected


# ── engineer_features: missing column guard ───────────────────────────────────

class TestInputValidation:
    def test_raises_if_begin_date_missing(self, df_clean):
        bad = df_clean.drop(columns=["begin_date"])
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer_features(bad)

    def test_end_date_absence_is_valid_for_live_inference(self, df_clean):
        """
        end_date is NOT required — its absence signals live inference mode.
        engineer_features must handle this gracefully (no churn target derived).
        """
        no_end = df_clean.drop(columns=["end_date"])
        result = engineer_features(no_end)
        assert "end_date"   not in result.columns
        assert "begin_date" not in result.columns
        assert "tenure_days" in result.columns
        assert "churn" not in result.columns  # no target in live mode
