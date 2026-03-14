# src/etl/ingest.py
"""
Raw data ingestion — loads and merges the four Interconnect source files.

This module is the single entry point for all raw data access.
It is called by:
  - The ETL pipeline (src/etl/transform.py)
  - The CI workflow (.github/workflows/ci.yml)
  - The notebook (Phase 1.2)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Source file registry ───────────────────────────────────────────────────────
_SOURCE_FILES = {
    "contract": "contract.csv",
    "personal": "personal.csv",
    "internet": "internet.csv",
    "phone":    "phone.csv",
}

_PRIMARY_KEY = "customerID"


def load_raw_data(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """
    Load and merge the four Interconnect source CSVs into one DataFrame.

    Parameters
    ----------
    raw_dir : str | Path
        Directory containing the four raw CSV files.
        Defaults to 'data/raw' (project root relative).

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with all customers and features.
        Shape: (7043, 20) for the full Interconnect dataset.

    Raises
    ------
    FileNotFoundError
        If any source CSV is missing from raw_dir.
    ValueError
        If the primary key (customerID) is missing or has duplicates
        in any source file.
    """
    raw_dir = Path(raw_dir)

    # ── Load each source file ──────────────────────────────────────────
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in _SOURCE_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Source file not found: {path}\n"
                f"Expected files: {list(_SOURCE_FILES.values())}"
            )
        frames[name] = pd.read_csv(path)
        logger.info("Loaded %s: %d rows, %d cols",
                    filename, *frames[name].shape)

    # ── Validate primary key in each source ────────────────────────────
    for name, df in frames.items():
        if _PRIMARY_KEY not in df.columns:
            raise ValueError(
                f"Primary key '{_PRIMARY_KEY}' not found in {name}.csv. "
                f"Columns found: {df.columns.tolist()}"
            )
        duplicates = df[_PRIMARY_KEY].duplicated().sum()
        if duplicates > 0:
            raise ValueError(
                f"{duplicates} duplicate customerIDs found in {name}.csv."
            )

    # ── Merge strategy: contract is the base (all customers) ──────────
    # personal  → INNER JOIN (every customer has personal data)
    # internet  → LEFT JOIN  (not all customers have internet service)
    # phone     → LEFT JOIN  (not all customers have phone service)
    df = (
        frames["contract"]
        .merge(frames["personal"], on=_PRIMARY_KEY, how="inner")
        .merge(frames["internet"], on=_PRIMARY_KEY, how="left")
        .merge(frames["phone"],    on=_PRIMARY_KEY, how="left")
    )

    # ── Post-merge assertions ──────────────────────────────────────────
    expected_rows = len(frames["contract"])
    if len(df) != expected_rows:
        raise ValueError(
            f"Row count changed after merge: "
            f"expected {expected_rows}, got {len(df)}. "
            f"Check for duplicate keys in source files."
        )

    duplicates_after = df[_PRIMARY_KEY].duplicated().sum()
    if duplicates_after > 0:
        raise ValueError(
            f"{duplicates_after} duplicate rows after merge. "
            f"Investigate join keys."
        )

    logger.info(
        "Merge complete: %d customers, %d features", *df.shape
    )
    return df


def get_source_summary(raw_dir: str | Path = "data/raw") -> dict:
    """
    Return a summary of each source file without merging.
    Useful for data quality audits and notebook reporting.

    Returns
    -------
    dict
        Keys: source names. Values: dict with shape, columns, null counts.
    """
    raw_dir = Path(raw_dir)
    summary = {}

    for name, filename in _SOURCE_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            summary[name] = {"error": f"File not found: {path}"}
            continue

        df = pd.read_csv(path)
        summary[name] = {
            "rows"    : len(df),
            "cols"    : len(df.columns),
            "columns" : df.columns.tolist(),
            "nulls"   : df.isnull().sum().to_dict(),
        }

    return summary
