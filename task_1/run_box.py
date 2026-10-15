from pathlib import Path
import subprocess
from tqdm import tqdm

from data.runs.utils import get_data

lmp = "/Applications/LAMMPS_GUI.app/Contents/bin/lmp" # put here your's lammps path
folder = Path(__file__).resolve().parent


def run_box(T, P, cells):
    """Reuse a completed run for this state and size, otherwise run LAMMPS."""
    cached = get_data([T], [P], [cells])
    if not cached.empty:
        return cached

    command = [
        lmp,
        "-log", "none",
        "-in", "box.lmp",
        "-var", "T", f"{T:g}",
        "-var", "P", f"{P:g}",
        "-var", "cells", f"{cells:g}",
    ]

    result = subprocess.run(
        command,
        cwd=folder,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout + result.stderr)
        result.check_returncode()

    data = get_data([T], [P], [cells])
    if data.empty:
        raise RuntimeError(f"LAMMPS produced no complete block data for T={T}, P={P}, cells={cells}")
    return data

def run_boxes(temperatures, pressures, cellses):
    """Load cached results, simulate missing combinations, and return all blocks."""
    with tqdm(total=len(temperatures) * len(pressures) * len(cellses), unit="run") as bar:
        for T in temperatures:
            for P in pressures:
                for cells in cellses:
                    bar.set_postfix(T=T, P=P, cells=4*cells**3)
                    run_box(T, P, cells)
                    bar.update(1)
    return get_data(temperatures, pressures, cellses)

if __name__ == "__main__":
    temperatures = [144]
    pressures = [1]
    cellses = [5]

    run_boxes(temperatures, pressures, cellses)
