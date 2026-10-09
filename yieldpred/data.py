"""Loading the reaction-yield dataset.

The default source is the CSV shipped in data/. MySQL is optional and is
configured only through environment variables, so no credentials live in code.
"""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "chemical_reaction_yield_dataset.csv"

FEATURES = ["temperature", "pressure", "catalyst_concentration"]
TARGET = "yield_pct"

# Physical yield cannot exceed 100 %; in this dataset 27 % of rows sit exactly
# on that ceiling (complete conversion), which is a censoring effect.
YIELD_CAP = 100.0

# Training domain. Predictions outside it are extrapolation and are flagged.
DOMAIN = {
    "temperature": (50.0, 300.0),          # deg C
    "pressure": (1.0, 10.0),               # atm
    "catalyst_concentration": (0.01, 0.50),  # mol/L
}

_CSV_COLUMNS = {
    "Temperature_C": "temperature",
    "Pressure_atm": "pressure",
    "Catalyst_Concentration_mol_L": "catalyst_concentration",
    "Reaction_Yield_percent": TARGET,
}


def load_csv(path=CSV_PATH):
    df = pd.read_csv(path).rename(columns=_CSV_COLUMNS)
    return _validate(df)


def load_mysql():
    """Read the `data` table described in init_experiments.sql.

    Set YIELD_DB_HOST, YIELD_DB_USER, YIELD_DB_PASSWORD and YIELD_DB_NAME.
    """
    import mysql.connector  # optional dependency

    db = mysql.connector.connect(
        host=os.environ.get("YIELD_DB_HOST", "localhost"),
        user=os.environ["YIELD_DB_USER"],
        password=os.environ["YIELD_DB_PASSWORD"],
        database=os.environ.get("YIELD_DB_NAME", "experiments"),
    )
    try:
        df = pd.read_sql("SELECT temperature, pressure, catalyst_concentration, "
                         "yield_pct FROM data", db)
    finally:
        db.close()
    return _validate(df)


def load(source=None):
    source = source or os.environ.get("YIELD_DATA_SOURCE", "csv")
    if source == "mysql":
        return load_mysql()
    if source == "csv":
        return load_csv()
    raise ValueError(f"unknown data source {source!r} (use 'csv' or 'mysql')")


def _validate(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    df = df[FEATURES + [TARGET]].astype(float)
    if df.isna().any().any():
        raise ValueError("dataset contains missing values")
    if (df[TARGET] < 0).any() or (df[TARGET] > YIELD_CAP + 1e-9).any():
        raise ValueError("yield outside 0-100 %")
    return df


def out_of_domain(temperature, pressure, catalyst_concentration):
    """Names of inputs that fall outside the training range."""
    values = {"temperature": temperature, "pressure": pressure,
              "catalyst_concentration": catalyst_concentration}
    return [k for k, v in values.items()
            if not DOMAIN[k][0] <= v <= DOMAIN[k][1]]
