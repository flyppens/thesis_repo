import numpy as np
from scipy.optimize import fsolve
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed

# --- 1. Constants from your snippet ---
tdp = 71
i_cf = 0.451
actual_op_carbon = 0.0004953

# --- 2. Base values (for centering your ranges) ---
p0 = 67436544      # parameter
t0 = 53793700        # token_count
lg_alpha0 = 35.839163 # log₁₀(alpha)

# --- 3. Wrapper that returns the error_rate for one sample ---
def model_error_rate(sample: np.ndarray) -> float:
    parameter, token_count, lg_alpha = sample

    b = 6 * parameter * token_count / 1e12
    alpha = np.exp(np.log(10)*lg_alpha)

    # 3. Define the same equation_a(x, a, b) from your script
    def equation_a(x):
        # np.log is natural log
        return np.log(1 + alpha * x) / alpha + x * np.log(1 + alpha * x) - x - b

    x_solution, = fsolve(equation_a, x0=1e8)

    # 5. Convert to energy (MWh) and carbon (tCO₂)
    energy_mwh = x_solution * tdp / 3600 / 1e3
    carbon_pred = energy_mwh * i_cf
    print(lg_alpha)
    print(carbon_pred)
    return (carbon_pred - actual_op_carbon) / actual_op_carbon

# --- 4. Define your SALib problem ---
problem = {
    "num_vars": 3,
    "names": ["param", "token", "lg_alpha"],
    "bounds": [
        [p0 * 0.94, p0 * 1.0279],
        [t0 * 0.98, t0 * 1.02],
        [lg_alpha0 * 0.16, lg_alpha0 * 3], 
    ],
}

# --- 5. Generate Saltelli samples ---
N = 1024  # base sample size (tunable)
X = saltelli.sample(problem, N, calc_second_order=True)

# --- 6. Evaluate in parallel ---
Y = Parallel(n_jobs=-1)(
    delayed(model_error_rate)(x) for x in X
)
Y = np.asarray(Y)

# --- 7. Compute Sobol indices ---
Si = sobol.analyze(
    problem,
    Y,
    calc_second_order=True,
    print_to_console=False
)

# --- 8. Print S1, S2, ST in a human-readable form ---
# First & total orders
print(f"\n{'Feature':<10} {'S1':>6} {'ST':>6}")
print("-" * 24)
for name, s1, st in zip(problem["names"], Si["S1"], Si["ST"]):
    print(f"{name:<10} {s1:6.3f} {st:6.3f}")

# Second orders
print(f"\n{'Pair':<20} {'S2':>6}")
print("-" * 28)
names = problem["names"]
for i in range(3):
    for j in range(i+1, 3):
        print(f"{names[i]} & {names[j]:<12} {Si['S2'][i,j]:6.3f}")
