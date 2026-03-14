# tests/conftest.py
"""
Session-scoped fixtures shared across the test suite.

Notes
-----
- test_clean_data.py and test_modeling_data.py define their own
  module-scoped `df` fixtures internally — no conflict here.
- These session fixtures are used by test_transform.py and test_predict.py
  when they need the processed data files directly.
"""
import os

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def df_clean():
    path = "data/processed/df_clean.parquet"
    assert os.path.exists(path), (
        "df_clean.parquet not found. Run notebook Phase 1 first."
    )
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def df_modeling():
    path = "data/processed/df_modeling.parquet"
    assert os.path.exists(path), (
        "df_modeling.parquet not found. Run notebook Phase 2 first."
    )
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def champion_artifact():
    """
    Session-scoped champion artifact — loaded once for the entire test run.
    Avoids reloading the model pkl on every test module.
    """
    from src.models.predict import load_champion
    return load_champion()
