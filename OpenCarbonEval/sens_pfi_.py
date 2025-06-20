# sensitivity_pfi_gpt.py

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# ----- 1) Baseline inputs from pfi_gpt.py -----
parameter_base = 67436544      # parameters
token_base     = 53793700      # training tokens
tdp            = 71            # W
i_base         = 0.451        # tCO2eq / MWh
lg_alpha_base  = 35.839163     # log10(alpha)
actual_carbon  = 0.4953    # tCO2eq  (ground truth)

# ----- 2) Sweep settings (same style as sens_analysis.py) -----
N_POINTS            = 15
#PARAM_RANGE_FRAC    = (0.94, 1.0279)
#TOKEN_RANGE_FRAC    = (0.98, 1.02)
#LG_ALPHA_RANGE_FRAC = (0.16, 3)
PARAM_RANGE_FRAC    = (0.5, 1.5)
TOKEN_RANGE_FRAC    = (0.5, 1.5)
LG_ALPHA_RANGE_FRAC = (0.5, 1.5)

# ----- 3) Your core carbon‐prediction function from pfi_gpt.py -----
def carbon_pred(p, t, lg_a):
    b     = 6 * p * t / 1e12
    alpha = np.exp(np.log(10)*lg_a)

    def equation_a(x):
        return (
            np.log(1 + alpha * x) / alpha
            + x * np.log(1 + alpha * x)
            - x
            - b
        )

    # fsolve returns a 1‐element array, so we unpack directly to a float
    x_s, = fsolve(equation_a, x0=1e8)

    # convert seconds → MWh → tCO₂
    energy_mwh = x_s * tdp / 3600 / 1e3
    carbon = energy_mwh * i_base
    print(x_s)
    return carbon

# ----- 4) Build the “row11‐style” table -----
rows = []

# 4a) parameter sweep
for p in np.linspace(parameter_base * PARAM_RANGE_FRAC[0],
                     parameter_base * PARAM_RANGE_FRAC[1],
                     N_POINTS):
    cp  = carbon_pred(p, token_base, lg_alpha_base)
    err = (actual_carbon-cp) / actual_carbon
    rows.append({
        "parameter # (B)" : p,
        "token   # (B)"   : token_base,
        "lg_alpha"        : lg_alpha_base,
        "carbon_pred_t"   : cp,        # tCO₂
        "error_rate"      : err,       # fractional
        "error_% "        : err * 100, # percent
    })

# 4b) token sweep
for t in np.linspace(token_base * TOKEN_RANGE_FRAC[0],
                     token_base * TOKEN_RANGE_FRAC[1],
                     N_POINTS):
    cp  = carbon_pred(parameter_base, t, lg_alpha_base)
    err = (actual_carbon-cp) / actual_carbon
    rows.append({
        "parameter # (B)" : parameter_base,
        "token   # (B)"   : t,
        "lg_alpha"        : lg_alpha_base,
        "carbon_pred_t"   : cp,
        "error_rate"      : err,
        "error_% "        : err * 100,
    })

# 4c) lg_alpha sweep
for la in np.linspace(lg_alpha_base * LG_ALPHA_RANGE_FRAC[0],
                      lg_alpha_base * LG_ALPHA_RANGE_FRAC[1],
                      N_POINTS):
    cp  = carbon_pred(parameter_base, token_base, la)
    err = (actual_carbon-cp) / actual_carbon
    rows.append({
        "parameter # (B)" : parameter_base,
        "token   # (B)"   : token_base,
        "lg_alpha"        : la,
        "carbon_pred_t"   : cp,
        "error_rate"      : err,
        "error_% "        : err * 100,
    })

# ----- 5) Dump to CSV just like sens_analysis does -----
out_df = pd.DataFrame(rows)
out_df.to_csv("sensitivity_pfi_gpt_row11.csv", index=False)
print(f"Saved {len(out_df)} rows → sensitivity_pfi_gpt_row11.csv")
