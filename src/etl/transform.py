# src/etl/transform.py
"""
ChurnPreprocessor — single source of truth for all data cleaning.

This transformer is the ONLY place where cleaning logic lives.
Used by: notebook Phase 1, FastAPI inference, CI pipeline, tests.

Design principles:
  - Stateless: fit() does nothing, all logic is deterministic
  - Fail-loud: special characters raise ValueError, never silent fix
  - Idempotent: running twice produces the same result
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SPECIAL_CHAR_PATTERN = r'[@#!?$%&*<>;|\"\']'

SERVICE_COLS = [
    "online_security",
    "internet_service",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "multiple_lines",
]

ABBREVIATION_MAP = {
    "ID" : "Id",
    "TV" : "Tv",
    "DSL": "Dsl",
}


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that cleans raw Interconnect data.

    Steps (in order):
      1. Standardize column names  PascalCase → snake_case
      2. Detect special characters  raise ValueError if found
      3. Strip leading/trailing whitespace from all string columns
      4. Cast types: dates → datetime64, TotalCharges → float64
      5. Impute TotalCharges empty strings → 0.0
      6. Impute service nulls (LEFT JOIN artifact) → 'No service'
      7. Encode SeniorCitizen 0/1 → 'No'/'Yes'
      8. Cast categorical columns → category dtype

    Parameters
    ----------
    None — transformer is fully stateless.

    Examples
    --------
    >>> from src.etl.ingest import load_raw_data
    >>> from src.etl.transform import ChurnPreprocessor
    >>> df_raw   = load_raw_data('data/raw')
    >>> df_clean = ChurnPreprocessor().fit_transform(df_raw)
    >>> df_clean.shape
    (7043, 20)
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnPreprocessor":
        """Stateless — nothing to learn. Returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full cleaning pipeline to raw DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Raw merged DataFrame from load_raw_data().

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame ready for feature engineering.

        Raises
        ------
        ValueError
            If special characters are detected in any string column.
        ValueError
            If required columns are missing after standardization.
        """
        df = X.copy()

        # Step 1 — column names
        df = self._standardize_columns(df)
        logger.info("Step 1 complete: columns standardized")

        # Step 2 — special character detection (fail loud)
        self._detect_special_chars(df)
        logger.info("Step 2 complete: no special characters detected")

        # Step 3 — whitespace
        df = self._strip_whitespace(df)
        logger.info("Step 3 complete: whitespace stripped")

        # Step 4 + 5 — type casting and imputation
        df = self._cast_types(df)
        logger.info("Step 4+5 complete: types cast, TotalCharges imputed")

        # Step 6 — service nulls from LEFT JOIN
        df = self._impute_service_nulls(df)
        logger.info("Step 6 complete: service nulls imputed")

        # Step 7 — senior_citizen encoding
        df = self._encode_senior_citizen(df)
        logger.info("Step 7 complete: senior_citizen encoded")

        # Step 8 — category dtype
        df = self._cast_categoricals(df)
        logger.info("Step 8 complete: categoricals cast")

        self._validate_output(df)
        logger.info("Output validation passed: %d rows, %d cols", *df.shape)

        return df

    # ── Private helpers ────────────────────────────────────────────────────────

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert PascalCase column names to snake_case."""
        df = df.copy()
        df.columns = [self._to_snake(col) for col in df.columns]
        return df

    @staticmethod
    def _to_snake(name: str) -> str:
        """
        Convert a single column name to snake_case.
        Handles known abbreviations (ID→Id, TV→Tv, DSL→Dsl) first.
        """
        for abbr, replacement in ABBREVIATION_MAP.items():
            name = name.replace(abbr, replacement)
        s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])",    r"\1_\2", s1)
        return s2.lower().replace(" ", "_").replace("-", "_")

    def _detect_special_chars(self, df: pd.DataFrame) -> None:
        """
        Scan all string columns for special characters.
        Raises ValueError immediately — never silently corrects.
        """
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            mask = (
                df[col]
                .astype(str)
                .str.contains(SPECIAL_CHAR_PATTERN, regex=True, na=False)
            )
            if mask.any():
                bad_values = df.loc[mask, col].unique().tolist()
                raise ValueError(
                    f"Special characters detected in column '{col}': "
                    f"{bad_values}.\n"
                    f"Review source data before proceeding. "
                    f"Automatic correction is disabled by design."
                )

    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all string columns."""
        df = df.copy()
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast dates and numeric columns to correct dtypes."""
        df = df.copy()

        # Dates
        df["begin_date"] = pd.to_datetime(df["begin_date"])
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", format="mixed")

        # TotalCharges: empty string → NaN → 0.0
        df["total_charges"] = pd.to_numeric(
            df["total_charges"], errors="coerce"
        ).fillna(0.0)

        return df

    def _impute_service_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN in service columns with 'No service'.
        These nulls arise from LEFT JOINs with internet/phone tables —
        they mean the customer does not have that service, not missing data.
        """
        df = df.copy()
        cols_present = [c for c in SERVICE_COLS if c in df.columns]
        for col in cols_present:
            df[col] = df[col].fillna("No service")
        return df

    def _encode_senior_citizen(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map SeniorCitizen 0/1 integer to 'No'/'Yes' string."""
        df = df.copy()
        if df["senior_citizen"].dtype != object:
            df["senior_citizen"] = (
                df["senior_citizen"].map({0: "No", 1: "Yes"})
            )
        return df

    def _cast_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast all object columns (except customer_id) to category dtype."""
        df = df.copy()
        cat_cols = (
            df.select_dtypes(include="object")
            .columns
            .difference(["customer_id"])
        )
        df[cat_cols] = df[cat_cols].astype("category")
        return df

    def _validate_output(self, df: pd.DataFrame) -> None:
        """
        Post-transform assertions.
        Raises AssertionError if any invariant is violated.
        """
        assert len(df) > 0, \
            "Output DataFrame is empty."
        assert "customer_id" in df.columns, \
            "customer_id missing from output."
        assert "begin_date" in df.columns, \
            "begin_date missing — required for feature engineering."
        assert str(df["begin_date"].dtype).startswith("datetime64"), \
            f"begin_date wrong dtype: {df['begin_date'].dtype}"
        assert df["total_charges"].dtype == "float64", \
            f"total_charges wrong dtype: {df['total_charges'].dtype}"

        # No nulls except end_date (NaT = active customer)
        cols_to_check = df.columns.difference(["end_date"])
        null_counts = df[cols_to_check].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, \
            f"Unexpected nulls after cleaning: {cols_with_nulls.to_dict()}"
