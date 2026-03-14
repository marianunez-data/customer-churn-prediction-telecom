# test_modeling_data.py
import json
import os

import great_expectations as gx
import pandas as pd
import pytest

MODELING_PATH = "data/processed/df_modeling.parquet"
REPORT_PATH   = "reports/ge_validation_features_ci.json"

LEAKAGE_COLS  = ["end_date", "begin_date"]


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(MODELING_PATH), (
        "df_modeling.parquet not found. Run notebook Phase 2 first."
    )
    return pd.read_parquet(MODELING_PATH)


@pytest.fixture(scope="module")
def ge_results(df):
    context = gx.get_context()

    datasource = context.data_sources.add_or_update_pandas(name="ci_modeling")

    try:
        asset = datasource.add_dataframe_asset(name="customers_modeling_ci")
    except Exception:
        asset = datasource.get_asset(name="customers_modeling_ci")

    batch_def = asset.add_batch_definition_whole_dataframe("full_batch")

    try:
        context.suites.delete("ci_modeling_suite")
    except Exception:
        pass
    suite = context.suites.add(gx.ExpectationSuite(name="ci_modeling_suite"))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="churn")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="churn", value_set=[0, 1]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnMeanToBeBetween(
            column="churn", min_value=0.05, max_value=0.50
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure_days")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure_days", min_value=0, max_value=3_650
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000, max_value=100_000
        )
    )

    try:
        context.validation_definitions.delete("ci_modeling_validation")
    except Exception:
        pass
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="ci_modeling_validation", data=batch_def, suite=suite
        )
    )

    results = validation_def.run(batch_parameters={"dataframe": df})
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results.to_json_dict(), f, indent=2)
    return results


class TestModelingData:
    def test_all_expectations_pass(self, ge_results):
        n_failed = ge_results.statistics["unsuccessful_expectations"]
        n_total  = ge_results.statistics["evaluated_expectations"]
        assert ge_results.success, (
            f"{n_failed}/{n_total} GE expectations failed. "
            f"See {REPORT_PATH} for details."
        )

    def test_no_leakage_columns(self, df):
        """Date and ID columns must not reach the feature matrix."""
        for col in LEAKAGE_COLS:
            assert col not in df.columns, (
                f"DATA LEAKAGE: '{col}' found in modeling dataset. "
                f"Drop it before saving df_modeling.parquet."
            )

    def test_target_is_binary(self, df):
        unique_vals = set(df["churn"].unique())
        assert unique_vals.issubset({0, 1}), (
            f"churn must be binary 0/1, found: {unique_vals}"
        )

    def test_no_nulls_in_features(self, df):
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"Nulls found in feature columns: {cols_with_nulls.to_dict()}"
        )

    def test_tenure_non_negative(self, df):
        assert (df["tenure_days"] >= 0).all(), (
            "tenure_days has negative values — check date engineering."
        )

    def test_report_saved(self):
        assert os.path.exists(REPORT_PATH), (
            f"GE report not found at {REPORT_PATH}"
        )
