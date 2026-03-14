# test_clean_data.py
import json
import os

import great_expectations as gx
import pandas as pd
import pytest

CLEAN_PATH  = "data/processed/df_clean.parquet"
REPORT_PATH = "reports/ge_validation_clean_ci.json"


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(CLEAN_PATH), (
        "df_clean.parquet not found. Run notebook Phase 1 first."
    )
    return pd.read_parquet(CLEAN_PATH)


@pytest.fixture(scope="module")
def ge_results(df):
    context = gx.get_context()

    # ── Datasource ─────────────────────────────────────────────────────
    datasource = context.data_sources.add_or_update_pandas(name="ci_clean")

    try:
        asset = datasource.add_dataframe_asset(name="customers_clean_ci")
    except Exception:
        asset = datasource.get_asset(name="customers_clean_ci")

    batch_def = asset.add_batch_definition_whole_dataframe("full_batch")

    # ── Suite — delete and recreate for idempotency ────────────────────
    try:
        context.suites.delete("ci_clean_suite")
    except Exception:
        pass
    suite = context.suites.add(gx.ExpectationSuite(name="ci_clean_suite"))

    # ── Expectations ───────────────────────────────────────────────────
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000, max_value=100_000
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToEqual(value=20)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="customer_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="monthly_charges", min_value=0, max_value=200
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_charges", min_value=0, max_value=15_000
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="type",
            value_set=["Month-to-month", "One year", "Two year"],
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="gender", value_set=["Male", "Female"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="begin_date", type_="datetime64"
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="end_date", type_="datetime64"
        )
    )

    # ── Validation definition ──────────────────────────────────────────
    try:
        context.validation_definitions.delete("ci_clean_validation")
    except Exception:
        pass
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="ci_clean_validation", data=batch_def, suite=suite
        )
    )

    results = validation_def.run(batch_parameters={"dataframe": df})
    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results.to_json_dict(), f, indent=2)
    return results

class TestCleanData:
    def test_all_expectations_pass(self, ge_results):
        n_failed = ge_results.statistics["unsuccessful_expectations"]
        n_total  = ge_results.statistics["evaluated_expectations"]
        assert ge_results.success, (
            f"{n_failed}/{n_total} GE expectations failed. "
            f"See {REPORT_PATH} for details."
        )

    def test_no_nulls_in_key_columns(self, df):
        key_cols = ["customer_id", "monthly_charges", "total_charges",
                    "type", "gender", "begin_date"]
        for col in key_cols:
            assert df[col].isnull().sum() == 0, (
                f"Unexpected nulls in '{col}': {df[col].isnull().sum()}"
            )

    def test_begin_date_before_cutoff(self, df):
        cutoff = pd.Timestamp("2020-02-01")
        assert (df["begin_date"] <= cutoff).all(), (
            "Found begin_date values after cutoff 2020-02-01"
        )

    def test_report_saved(self):
        assert os.path.exists(REPORT_PATH), (
            f"GE report not found at {REPORT_PATH}"
        )
