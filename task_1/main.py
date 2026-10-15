import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import product

    from data.utils import load_data
    from run_box import run_boxes
    from find_params import main as find_params
    from find_optim_size import run_study, get_ideal_params, base

    return (
        find_params,
        get_ideal_params,
        load_data,
        mo,
        pd,
        plt,
        product,
        run_boxes,
        run_study,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lennard-Jones parameters
    """)
    return


@app.cell(hide_code=True)
def _(find_params, mo):
    epsilon, sigma, _ = find_params()

    mo.hstack([mo.stat(f"{epsilon:.6f}", label="ε", caption="kcal/mol"),
               mo.stat(f"{sigma:.6f}", label="σ", caption="Å")]),
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## System size
    Time is equilibration + production in seconds per run; precision is the density tolerance in percent.
    """)
    return


@app.cell
def _(mo):
    max_time = mo.ui.number(value=2, label="Max time computation")
    precision = mo.ui.number(value=1, label="Density tolerance in percent")

    mo.hstack([max_time, precision])
    return max_time, precision


@app.cell(hide_code=True)
def _(get_ideal_params, max_time, mo, plt, precision, run_study):
    _summary = run_study()

    _params = get_ideal_params(max_time.value, precision.value)
    mo.stop(
        _params is None,
        mo.callout("No tested size meets both limits.", kind="warn"),
    )
    N, cells = _params["N"], _params["cells"]

    _chosen = N if N else None
    mo.hstack([mo.stat(N, label="Particles N"),
               mo.stat(cells, label="Cells per edge")])

    _table = _summary[["target_T", "target_P", "N", "rho_mean", "rho_ci95", "md_time_s"]].rename(columns={
        "target_T": "T (K)", "target_P": "P (atm)", "rho_mean": "ρ (g/ml)",
        "rho_ci95": "95% CI ± (g/ml)", "md_time_s": "MD time (s)",
    })
    mo.ui.table(_table.round(6),
                selection=None, pagination=False, show_data_types=False, label="Size study")

    _fig, _axes = plt.subplots(1, 2, figsize=(7, 3.3), layout="constrained")
    for (T, P), group in _summary.groupby(["target_T", "target_P"]):
        label = f"T={T:g} K, P={P:g} atm"
        _axes[0].errorbar(group["N"], group["rho_mean"], yerr=group["rho_ci95"], marker="o", capsize=3, label=label)
        _axes[1].plot(group["N"], group["md_time_s"], "o-", label=label)
    _axes[0].set(title="Density: approximate 95% interval", ylabel="Density (g/ml)")
    _axes[1].set(title="Equilibration + production", ylabel="MD time (s)")
    for ax in _axes:
        ax.set(xscale="log", xlabel="Number of particles N")
        ax.grid(True, alpha=0.3)
        if _chosen is not None:
            ax.axvline(_chosen, color="green", linestyle="--", label=f"Selected N={_chosen}")
        ax.legend(fontsize=8)

    mo.mpl.interactive(_fig)
    return (cells,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Isotherms
    """)
    return


@app.cell(hide_code=True)
def _(cells, load_data, pd, product, run_boxes):
    def ideal(pressures, temperatures):
        M = 0.0379968
        R = 8.314462618

        df = pd.DataFrame(
            product(pressures, temperatures),
            columns=["pressure", "temperature"]
        )
        # Convert pressure from atm to Pa and density from kg/m³ to g/ml.
        df["density"] = df["pressure"] * 101325 * M / (R * df["temperature"]) / 1000
        return df

    _temperatures = [120, 144, 160]
    _pressures = [1, 2, 3, 4, 5, 6, 8, 10]
    _cellses = [cells]

    df_simulated = run_boxes(_temperatures, _pressures, _cellses)
    df_real = load_data(temperatures=_temperatures, pressures=_pressures)
    df_ideal = ideal(_pressures, _temperatures)
    return df_ideal, df_real, df_simulated


@app.cell
def _(df_ideal, df_real, df_simulated, mo, plt):
    # Average the time blocks to obtain one point per simulation.
    _points = df_simulated.groupby("file", as_index=False)[["P", "rho"]].mean()
    _points["temperature"] = _points["file"].str.extract(r"^T([^_]+)", expand=False).astype(float)

    _fig, _ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    for _i, (_T, _group) in enumerate(_points.groupby("temperature")):
        _color = f"C{_i}"
        _group = _group.sort_values("P")
        _real = df_real[df_real["T_K"] == _T].sort_values("Pressure (atm)")
        _ideal = df_ideal[df_ideal["temperature"] == _T].sort_values("pressure")
        _ax.plot(_group["rho"], _group["P"], "o-", color=_color, label=f"{_T:g} K · simulated")
        _ax.plot(_real["Density (g/ml)"], _real["Pressure (atm)"], "s--", color=_color,
                 markersize=4, markerfacecolor="none", label=f"{_T:g} K · NIST")
        _ax.plot(_ideal["density"], _ideal["pressure"], ":", color=_color, label=f"{_T:g} K · ideal")
    _ax.set(title="Isotherms: simulation, NIST and ideal gas",
            xlabel="Density ρ (g/ml)", ylabel="Pressure P (atm)")
    _ax.grid(True, alpha=0.3)
    _ax.legend(ncols=3, fontsize=8, loc="upper left")
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
