# src/features/engineer.py
"""
Feature engineering for the Interconnect churn model.

Responsibilities:
  - Create binary churn target from end_date
  - Create tenure_days from begin_date and reference_date
  - Segment customers for production scoring
  - Drop date columns after feature extraction (leakage prevention)

Called by:
  - Notebook (§2.x feature engineering)
  - src/models/predict.py CLI (batch scoring)
  - Streamlit app (live inference with reference_date=today)
  - CI pipeline

Key design decision — reference_date parameter:
  - Historical data  → reference_date=CUTOFF_DATE (2020-02-01) — default
  - Live scoring     → reference_date=pd.Timestamp.today()
  - This single parameter makes the function correct in both contexts
    without any code duplication.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CUTOFF_DATE = pd.Timestamp("2020-02-01")   # dataset snapshot date
OBS_START   = pd.Timestamp("2019-10-01")   # first observable churn event

COLS_TO_DROP_MODELING = [
    "begin_date",    # used to compute tenure_days → drop after (leakage)
    "end_date",      # used to compute churn target → drop after (leakage)
    "tenure_months", # redundant with tenure_days if present
    "tenure_range",  # redundant binned version if present
]


def engineer_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    drop_segment: bool = False,
) -> pd.DataFrame:
    """
    Transform clean DataFrame into modeling-ready DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ChurnPreprocessor.fit_transform().
        Must contain: begin_date, end_date (or neither for live data),
        customer_id.

    reference_date : pd.Timestamp | None
        Date to compute tenure_days against.

        - None / not passed  → uses CUTOFF_DATE (2020-02-01).
          Correct for the historical Interconnect dataset.
        - pd.Timestamp.today() → use for live inference in production
          or Streamlit app when scoring new customers.

        This is the ONLY parameter that changes between training
        and live scoring. Everything else is identical.

    drop_segment : bool
        If True, drops the 'segment' column from output.
        Useful when segment is not a model feature (default: False).

    Returns
    -------
    pd.DataFrame
        Modeling-ready DataFrame with churn target (if end_date present)
        and tenure_days. Dates are dropped. customer_id retained.

    Raises
    ------
    ValueError
        If required input columns are missing.

    Examples
    --------
    # Historical training data:
    >>> df_modeling = engineer_features(df_clean)

    # Live Streamlit scoring (new customers, today's date):
    >>> df_modeling = engineer_features(
    ...     df_clean,
    ...     reference_date=pd.Timestamp.today(),
    ... )
    """
    _validate_input(df)

    ref_date = reference_date if reference_date is not None else CUTOFF_DATE
    df = df.copy()

    # ── Step 1: churn target ───────────────────────────────────────────
    # Only derived when end_date is present (training / evaluation).
    # For live inference, end_date does not exist → churn column skipped.
    if "end_date" in df.columns:
        df["churn"] = df["end_date"].notna().astype("int8")
        logger.info(
            "Churn target derived: %d churned / %d retained",
            df["churn"].sum(),
            (df["churn"] == 0).sum(),
        )
    else:
        logger.info(
            "end_date not present — churn target skipped (live inference mode)."
        )

    # ── Step 2: tenure_days ────────────────────────────────────────────
    # Active customers (end_date = NaT): tenure = begin_date → ref_date
    # Churned customers:                 tenure = begin_date → end_date
    # Live inference (no end_date):      tenure = begin_date → ref_date
    if "end_date" in df.columns:
        end_or_ref = df["end_date"].fillna(ref_date)
    else:
        end_or_ref = ref_date

    df["tenure_days"] = (end_or_ref - df["begin_date"]).dt.days

    # Guard against negative tenure (data quality issue)
    negative_mask = df["tenure_days"] < 0
    if negative_mask.any():
        n_neg = negative_mask.sum()
        logger.warning(
            "%d customers have negative tenure_days. "
            "Clamping to 0. Check begin_date vs reference_date.",
            n_neg,
        )
        df.loc[negative_mask, "tenure_days"] = 0

    logger.info(
        "tenure_days derived: min=%d, max=%d, mean=%.1f (ref_date=%s)",
        df["tenure_days"].min(),
        df["tenure_days"].max(),
        df["tenure_days"].mean(),
        ref_date.date(),
    )

    # ── Step 3: production segment ────────────────────────────────────
    # eligible → sufficient observation window for model scoring
    # new      → recently acquired, churn pattern not yet established
    df["segment"] = df["begin_date"].apply(
        lambda d: "eligible" if d < OBS_START else "new"
    )
    logger.info(
        "Segments: eligible=%d, new=%d",
        (df["segment"] == "eligible").sum(),
        (df["segment"] == "new").sum(),
    )

    # ── Step 4: drop date columns (leakage prevention) ────────────────
    cols_to_drop = [c for c in COLS_TO_DROP_MODELING if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    if drop_segment and "segment" in df.columns:
        df = df.drop(columns=["segment"])

    # ── Step 5: validate output ────────────────────────────────────────
    _validate_output(df)

    logger.info(
        "Feature engineering complete: %d rows, %d cols (ref_date=%s)",
        *df.shape,
        ref_date.date(),
    )
    return df


# ── Single-customer helpers (used by Streamlit & predict.py) ─────────────────

def compute_tenure(
    begin_date: pd.Timestamp,
    reference_date: pd.Timestamp | None = None,
) -> int:
    """
    Compute tenure_days for a single customer.

    Parameters
    ----------
    begin_date     : pd.Timestamp — customer acquisition date
    reference_date : pd.Timestamp — date to measure against.
        Default: CUTOFF_DATE (2020-02-01) for historical data.
        Pass pd.Timestamp.today() for live Streamlit scoring.

    Returns
    -------
    int : days since begin_date (clamped to 0 if negative)

    Examples
    --------
    # Historical:
    >>> compute_tenure(pd.Timestamp("2019-01-01"))
    397

    # Live:
    >>> compute_tenure(pd.Timestamp("2023-06-01"),
    ...                reference_date=pd.Timestamp.today())
    """
    ref = reference_date if reference_date is not None else CUTOFF_DATE
    days = int((ref - begin_date).days)
    return max(days, 0)


def classify_segment(begin_date: pd.Timestamp) -> str:
    """
    Classify a single customer as 'eligible' or 'new'.

    Parameters
    ----------
    begin_date : pd.Timestamp — customer acquisition date

    Returns
    -------
    str : 'eligible' if begin_date < OBS_START, else 'new'
    """
    return "eligible" if begin_date < OBS_START else "new"


# ── Private helpers ────────────────────────────────────────────────────────────

def _validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    required = {"begin_date", "customer_id"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for feature engineering: {missing}. "
            f"Run ChurnPreprocessor first."
        )
    # begin_date must be datetime
    if not pd.api.types.is_datetime64_any_dtype(df["begin_date"]):
        raise ValueError(
            f"begin_date must be datetime64, got {df['begin_date'].dtype}. "
            f"Run ChurnPreprocessor first."
        )


def _validate_output(df: pd.DataFrame) -> None:
    """Raise AssertionError if modeling DataFrame invariants are violated."""

    # Leakage checks
    if "end_date" in df.columns:
        raise AssertionError(
            "LEAKAGE: end_date present in modeling DataFrame. "
            "It must be dropped after deriving churn and tenure_days."
        )
    if "begin_date" in df.columns:
        raise AssertionError(
            "LEAKAGE: begin_date present in modeling DataFrame. "
            "It must be dropped after deriving tenure_days."
        )

    # Tenure integrity
    if "tenure_days" not in df.columns:
        raise AssertionError("tenure_days column missing from output.")
    if (df["tenure_days"] < 0).any():
        raise AssertionError(
            "tenure_days has negative values after clamping — check input dates."
        )

    # Churn integrity (only when derived)
    if "churn" in df.columns:
        if not df["churn"].isin([0, 1]).all():
            raise AssertionError("churn must be binary 0/1.")

    # No unexpected nulls
    null_counts     = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise AssertionError(
            f"Unexpected nulls in modeling DataFrame: "
            f"{cols_with_nulls.to_dict()}"
        )
