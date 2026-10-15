from math import hypot
from pathlib import Path
import re

import pandas as pd
from scipy.stats import t

from run_box import run_boxes
from data.runs.utils import base as runs_path

base = Path(__file__).resolve().parent
summary_path = base / "data" / "tmp" / "size_study.csv"

temperatures = [144]
pressures = [2]
cellses = [1, 2, 3, 4, 5, 6, 7, 8]


def run_study():
    """Reuse completed runs, simulate missing sizes, save the summary, and return it."""
    df = run_boxes(temperatures, pressures, cellses)
    summary = summarize_runs(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def summarize_runs(df):
    """Calculate density uncertainty, drift, finite-size agreement, and MD time."""

    # One result row per simulation. Each input row is already a time-block average.
    rows = []
    for filename, blocks in df.groupby("file", sort=False):
        parts = filename.split("_")
        target_T, target_P, N = float(parts[0][1:]), float(parts[1][1:]), int(parts[2][1:])
        count = len(blocks)
        if count < 2:
            raise ValueError(f"Need at least two time blocks in {filename}")

        # Approximate 95% confidence interval of the mean, assuming independent blocks.
        # Student's t accounts for having only ten blocks in the default input script.
        rho_mean = blocks["rho"].mean()
        rho_ci95 = t.ppf(0.975, count - 1) * blocks["rho"].sem()
        half = count // 2
        drift = abs(blocks["rho"].iloc[:half].mean() - blocks["rho"].iloc[half:].mean())

        # Both run commands print a Loop time: equilibration + production, in seconds.
        log = (runs_path / filename.replace(".blocks.dat", ".log")).read_text()
        times = re.findall(r"^Loop time of ([\d.eE+-]+)", log, flags=re.MULTILINE)
        if len(times) != 2:
            raise ValueError(f"Expected two MD run times for {filename}, found {len(times)}")

        rows.append({
            "target_T": target_T, "target_P": target_P,
            "cells": round((N / 4) ** (1 / 3)), "N": N, "blocks": count,
            "T_mean": blocks["T"].mean(), "P_mean": blocks["P"].mean(),
            "rho_mean": rho_mean, "rho_ci95": rho_ci95,
            "rho_ci95_percent": 100 * rho_ci95 / rho_mean,
            "drift_percent": 100 * drift / rho_mean,
            "md_time_s": sum(float(value) for value in times), "file": filename,
        })

    summary = pd.DataFrame(rows).sort_values(["target_T", "target_P", "N"]).reset_index(drop=True)
    summary["difference_bound_percent"] = 0.0

    # At each T and P, use the largest tested system as a finite-size reference.
    # Include uncertainty in the comparison, rather than comparing only two means.
    for _, group in summary.groupby(["target_T", "target_P"]):
        reference = group.iloc[-1]
        for index, row in group.iterrows():
            if row["N"] == reference["N"]:
                continue
            difference = abs(row["rho_mean"] - reference["rho_mean"])
            margin = hypot(row["rho_ci95"], reference["rho_ci95"])
            summary.loc[index, "difference_bound_percent"] = 100 * (difference + margin) / reference["rho_mean"]

    return summary


def get_ideal_params(max_time, precision):
    """Return {N, cells} for the smallest qualifying size, or None if none qualifies.

    Uses data/tmp/size_study.csv saved by run_study(). max_time limits equilibration +
    production time per run in seconds; precision limits relative density
    uncertainty, finite-size disagreement, and drift in percent (1 means 1%).
    The result applies only to the tested states and sampling lengths.
    """
    if not max_time > 0 or not precision > 0:
        raise ValueError("max_time (seconds) and precision (percent) must be positive")

    summary = pd.read_csv(summary_path)
    accuracy = ["rho_ci95_percent", "difference_bound_percent", "drift_percent"]
    passes = summary[accuracy].le(precision).all(axis=1)

    # Require every tested size at or above the candidate to pass at every test state.
    # The largest size alone cannot establish convergence, so it is not a candidate.
    # The time limit applies to the candidate, not the larger reference systems.
    sizes = sorted(summary["N"].unique())
    for N in sizes[:-1]:
        if passes[summary["N"] >= N].all() and summary.loc[summary["N"] == N, "md_time_s"].le(max_time).all():
            return {"N": int(N), "cells": round((N / 4) ** (1 / 3))}
    return None


if __name__ == "__main__":
    run_study()

