from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

base = Path(__file__).resolve().parent

def load_data(temperatures=None, pressures=None):
    """Load valid gas/vapor data, optionally matching temperatures (K) and pressures (atm).

    Each filter accepts a single value or a list; None keeps all values.
    Matches allow small rounding differences in the recorded NIST values.
    """
    M_F2 = 37.9968       # g/mol
    ATM_TO_PA = 101325.0

    folder = base / "fluorine"

    # ----------------------------------------
    # Read all NIST files
    # ----------------------------------------

    frames = []

    files = [file for file in folder.iterdir() if file.is_file()]
    for file in tqdm(files, desc="Loading NIST files", unit="file"):

        try:
            temp_df = pd.read_csv(file, sep="\t")
            frames.append(temp_df)
        except Exception as e:
            tqdm.write(f"Could not load {file.name}: {e}")

    df = pd.concat(
        frames,
        ignore_index=True
    )

    for column, values in [("Temperature (K)", temperatures), ("Pressure (atm)", pressures)]:
        if values is not None:
            matches = np.isclose(
                df[column].to_numpy()[:, None], np.atleast_1d(values),
                rtol=1e-5, atol=1e-8,
            ).any(axis=1)
            df = df[matches]

    # ----------------------------------------
    # Convert NIST units
    # ----------------------------------------

    df["T_K"] = df["Temperature (K)"]

    df["P_Pa"] = (
        df["Pressure (atm)"]
        * ATM_TO_PA
    )

    df["rho_mol_m3"] = (
        df["Density (g/ml)"]
        * 1e6
        / M_F2
    )


    # ----------------------------------------
    # Keep only valid data
    # ----------------------------------------

    df = df[
        np.isfinite(df["T_K"])
        & np.isfinite(df["P_Pa"])
        & np.isfinite(df["rho_mol_m3"])
    ]

    df = df[
        (df["P_Pa"] > 0)
        & (df["rho_mol_m3"] > 0)
    ]

    df = df[
        df["Phase"]
        .astype(str)
        .str.lower()
        .isin(["vapor", "gas"])
    ]

    return df
