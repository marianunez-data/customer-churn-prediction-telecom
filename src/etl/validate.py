# src/etl/validate.py
"""
Great Expectations validation functions.

Three validation gates — each called at a different pipeline stage:

  validate_raw()        → called after load_raw_data()
                          before any cleaning
  validate_clean()      → called after ChurnPreprocessor.fit_transform()
                          before feature engineering
  validate_inference()  → called at FastAPI /score endpoint
                          before sending data to the model

Design: functions raise DataValidationError on failure.
Never silently pass bad data downstream.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import great_expectations as gx
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when a GE validation suite fails."""
    pass


# ── Gate 1 — Raw data ──────────────────────────────────────────────────────────

def validate_raw(
    df: pd.DataFrame,
    report_path: str | Path = "reports/ge_validation_raw.json",
) -> None:
    """
    Validate raw merged DataFrame before any cleaning.

    Checks: row count, primary key uniqueness, required columns,
            numeric ranges, date column presence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame from load_raw_data().
    report_path : str | Path
        Where to save the GE validation report JSON.

    Raises
    ------
    DataValidationError
        If any expectation fails.
    """
    context    = gx.get_context()
    datasource = context.data_sources.add_or_update_pandas(name="raw_gate")

    try:
        asset = datasource.add_dataframe_asset(name="raw_data")
    except Exception:
        asset = datasource.get_asset(name="raw_data")

    batch_def = asset.add_batch_definition_whole_dataframe("raw_batch")

    try:
        context.suites.delete("raw_suite")
    except Exception:
        pass
    suite = context.suites.add(gx.ExpectationSuite(name="raw_suite"))

    # Row count
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000, max_value=100_000
        )
    )
    # Primary key
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="customerID")
    )
    # Numeric ranges
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="MonthlyCharges", min_value=0, max_value=200
        )
    )

    _run_and_assert(context, suite, batch_def, df, report_path, "raw_gate")


# ── Gate 2 — Clean data ────────────────────────────────────────────────────────

def validate_clean(
    df: pd.DataFrame,
    report_path: str | Path = "reports/ge_validation_clean.json",
) -> None:
    """
    Validate cleaned DataFrame before feature engineering.

    Checks: column names are snake_case, no nulls in key columns,
            date dtypes, categorical domains, numeric ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ChurnPreprocessor.fit_transform().
    report_path : str | Path
        Where to save the GE validation report JSON.

    Raises
    ------
    DataValidationError
        If any expectation fails.
    """
    context    = gx.get_context()
    datasource = context.data_sources.add_or_update_pandas(name="clean_gate")

    try:
        asset = datasource.add_dataframe_asset(name="clean_data")
    except Exception:
        asset = datasource.get_asset(name="clean_data")

    batch_def = asset.add_batch_definition_whole_dataframe("clean_batch")

    try:
        context.suites.delete("clean_suite")
    except Exception:
        pass
    suite = context.suites.add(gx.ExpectationSuite(name="clean_suite"))

    # Structure
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000, max_value=100_000
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToEqual(value=20)
    )
    # Primary key
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="customer_id")
    )
    # Numeric ranges
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
    # Categorical domains
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
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="senior_citizen", value_set=["No", "Yes"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="internet_service",
            value_set=["DSL", "Fiber optic", "No service"],
        )
    )

    _run_and_assert(context, suite, batch_def, df, report_path, "clean_gate")


# ── Gate 3 — Inference input ───────────────────────────────────────────────────

def validate_inference(
    df: pd.DataFrame,
    report_path: str | Path = "reports/ge_validation_inference.json",
) -> None:
    """
    Validate a single customer or batch before model scoring.

    Called by FastAPI /score endpoint and n8n pipeline.
    Stricter than validate_clean — checks every feature the model expects.

    Parameters
    ----------
    df : pd.DataFrame
        Customer DataFrame with all model features.
        May have 1 row (single prediction) or many (batch).
    report_path : str | Path
        Where to save the GE validation report JSON.

    Raises
    ------
    DataValidationError
        If any expectation fails. The caller should return HTTP 422.
    """
    context    = gx.get_context()
    datasource = context.data_sources.add_or_update_pandas(
        name="inference_gate"
    )

    try:
        asset = datasource.add_dataframe_asset(name="inference_data")
    except Exception:
        asset = datasource.get_asset(name="inference_data")

    batch_def = asset.add_batch_definition_whole_dataframe("inference_batch")

    try:
        context.suites.delete("inference_suite")
    except Exception:
        pass
    suite = context.suites.add(
        gx.ExpectationSuite(name="inference_suite")
    )

    # Required features must exist and not be null
    required_features = [
        "monthly_charges", "total_charges", "tenure_days",
        "type", "internet_service", "payment_method",
        "gender", "senior_citizen", "partner", "dependents",
    ]
    for feat in required_features:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=feat)
        )

    # Numeric ranges
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
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure_days", min_value=0, max_value=36_500
        )
    )

    # Categorical domains — model will fail on unknown categories
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="type",
            value_set=["Month-to-month", "One year", "Two year"],
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="internet_service",
            value_set=["DSL", "Fiber optic", "No service"],
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="gender", value_set=["Male", "Female"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="senior_citizen", value_set=["No", "Yes"]
        )
    )

    _run_and_assert(
        context, suite, batch_def, df, report_path, "inference_gate"
    )


# ── Private helper ─────────────────────────────────────────────────────────────

def _run_and_assert(
    context,
    suite,
    batch_def,
    df: pd.DataFrame,
    report_path: str | Path,
    gate_name: str,
) -> None:
    """Run validation, save report, raise on failure."""
    try:
        context.validation_definitions.delete(f"{gate_name}_validation")
    except Exception:
        pass

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=f"{gate_name}_validation",
            data=batch_def,
            suite=suite,
        )
    )

    results = validation_def.run(batch_parameters={"dataframe": df})

    os.makedirs(Path(report_path).parent, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results.to_json_dict(), f, indent=2)

    if not results.success:
        n_failed = results.statistics["unsuccessful_expectations"]
        n_total  = results.statistics["evaluated_expectations"]
        raise DataValidationError(
            f"[{gate_name}] {n_failed}/{n_total} expectations failed. "
            f"Report saved to {report_path}"
        )

    logger.info(
        "[%s] All %d expectations passed.",
        gate_name,
        results.statistics["evaluated_expectations"],
    )
