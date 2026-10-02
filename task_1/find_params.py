import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares
from tqdm import tqdm

R = 8.314462618          # J / (mol K)
N_A = 6.02214076e23      # 1 / mol
J_TO_KCAL = 1 / 4184
ANGSTROM = 1e-10


# ============================================================
# 1. Experimental B2(T)
# ============================================================

def get_B2_experimental(df, max_points=8, polynomial_order=2):
    """
    Estimate B2(T) from low-density experimental P-rho data.

    Required columns:
        T_K         temperature [K]
        P_Pa        pressure [Pa]
        rho_mol_m3  molar density [mol/m^3]

    Virial equation:
        Z = P/(rho R T)
          = 1 + B2*rho + B3*rho^2 + ...

    Returns:
        DataFrame with T_K and B2_exp_m3_mol.
    """

    result = []

    groups = df.groupby("T_K")
    for T, group in tqdm(groups, total=groups.ngroups,
                         desc="Experimental B2", unit="temperature"):

        # Sort by density and keep lowest-density points
        group = group.sort_values("rho_mol_m3").head(max_points)

        rho = group["rho_mol_m3"].to_numpy()
        P = group["P_Pa"].to_numpy()

        Z = P / (rho * R * T)

        # We fit:
        # Z - 1 = B2*rho + B3*rho^2 + ...
        #
        # No constant term because Z -> 1 when rho -> 0.

        X = np.column_stack([
            rho**n
            for n in range(1, polynomial_order + 1)
        ])

        coefficients, *_ = np.linalg.lstsq(
            X,
            Z - 1,
            rcond=None
        )

        B2 = coefficients[0]

        result.append({
            "T_K": T,
            "B2_exp_m3_mol": B2
        })

    return pd.DataFrame(result)


# ============================================================
# 2. Reduced second virial coefficient for LJ
# ============================================================

def B2_star(T_star):
    """
    Dimensionless second virial coefficient for
    Lennard-Jones 12-6 potential.

    B2* = -2*pi Integral[
        ( exp[-4/T* (x^-12 - x^-6)] - 1 ) x^2 dx
    ]
    """

    def integrand(x):

        if x == 0:
            return 0.0

        potential_reduced = 4.0 * (
            x**(-12) - x**(-6)
        )

        return (
            np.exp(-potential_reduced / T_star) - 1.0
        ) * x**2

    integral, error = quad(
        integrand,
        0,
        np.inf,
        epsabs=1e-9,
        epsrel=1e-8,
        limit=300
    )

    return -2 * np.pi * integral


# ============================================================
# 3. B2 predicted by Lennard-Jones
# ============================================================

def B2_LJ(T, epsilon_kcal_mol, sigma_A):
    """
    T                [K]
    epsilon_kcal_mol [kcal/mol]
    sigma_A          [Angstrom]

    Returns:
        B2 [m^3/mol]
    """

    epsilon_J_mol = epsilon_kcal_mol * 4184
    sigma_m = sigma_A * ANGSTROM

    # T* = R*T / epsilon
    T_star = R * T / epsilon_J_mol

    return (
        N_A
        * sigma_m**3
        * B2_star(T_star)
    )


# ============================================================
# 4. Fit epsilon and sigma
# ============================================================

def fit_LJ(B2_data,
           epsilon_initial=0.2,
           sigma_initial=3.5):

    temperatures = B2_data["T_K"].to_numpy()
    B2_exp = B2_data["B2_exp_m3_mol"].to_numpy()

    # Optimize logarithms so epsilon > 0 and sigma > 0
    x0 = np.log([
        epsilon_initial,
        sigma_initial
    ])

    # Characteristic scale to make residuals numerically nice
    scale = np.max(np.abs(B2_exp))

    def residuals(log_params):

        epsilon, sigma = np.exp(log_params)

        B2_model = np.array([
            B2_LJ(T, epsilon, sigma)
            for T in temperatures
        ])

        progress.update(1)
        return (B2_model - B2_exp) / scale

    # The optimizer's evaluation count is not known in advance.
    with tqdm(desc="Fitting LJ", unit="evaluation") as progress:
        result = least_squares(
            residuals,
            x0
        )

    epsilon, sigma = np.exp(result.x)

    return epsilon, sigma, result


# ============================================================
# 5. Example usage
# ============================================================

if __name__ == "__main__":
    from pathlib import Path

    M_F2 = 37.9968       # g/mol
    ATM_TO_PA = 101325.0

    folder = Path("task_1/data/fluorine")

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


    # Optionally keep only gas/vapor phase
    if "Phase" in df.columns:
        df = df[
            df["Phase"]
            .astype(str)
            .str.lower()
            .isin(["vapor", "gas"])
        ]


    # ----------------------------------------
    # Calculate experimental B2
    # ----------------------------------------

    B2_data = get_B2_experimental(
        df,
        max_points=8,
        polynomial_order=2
    )


    # ----------------------------------------
    # Fit Lennard-Jones parameters
    # ----------------------------------------

    epsilon, sigma, result = fit_LJ(
        B2_data,
        epsilon_initial=0.2,
        sigma_initial=3.5
    )

    print("\nLennard-Jones parameters:")
    print(f"epsilon = {epsilon:.6f} kcal/mol")
    print(f"sigma   = {sigma:.6f} AngsPhasestrom")
