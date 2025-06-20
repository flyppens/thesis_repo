"""
feature_importance_loglin.py
────────────────────────────
Reads the one-at-a-time sensitivity grid you saved as
`sensitivity_row11.csv`, fits a standardised log–linear model, prints the
β-coefficients (elasticities) and shows a bar-chart of their absolute
magnitudes.

Refs in comments:
  • Strubell et al., 2019 – ‘Energy and Policy Considerations for Deep Learning’
  • Patterson et al., 2021 – ‘Carbon Emissions and Large Neural Network Training’
  • Saltelli et al., 2008 – ‘Global Sensitivity Analysis: The Primer’
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from scipy.stats import shapiro



CSV = Path("sensitivity_row11.csv")      # adjust if stored elsewhere
df  = pd.read_csv(CSV)


cols_numeric = ["parameter # (B)", "token # (B)", "hardware efficiency",
                "pred_CO2eq_t", "train_delta", "error_% "]
df[cols_numeric] = df[cols_numeric].apply(pd.to_numeric, errors="coerce")


df = df.dropna(subset=cols_numeric).reset_index(drop=True)


df["log_error_% "]   = np.log(df["error_% "]*-1)
df["log_param"] = np.log(df["parameter # (B)"])
df["log_token"] = np.log(df["token # (B)"])
df["log_eff"]   = np.log(df["hardware efficiency"])


X_raw = df[["log_token", "log_param", "log_eff"]]
X     = (X_raw - X_raw.mean()) / X_raw.std(ddof=0)   # z-scores
y     = df["error_% "]
df['y_log'] = np.sign(y) * np.log1p(np.abs(y))
y_log = df['y_log']


lin   = LinearRegression().fit(X, y_log)
beta  = pd.Series(lin.coef_, index=X.columns)
X_sm = sm.add_constant(X)
lin2 = sm.OLS(y_log, X_sm)
results = lin2.fit()
#print(results.summary())
print(results.summary().as_latex())


print("\nStandardised β-coefficients (elasticities):")
print(beta.sort_values(key=abs, ascending=False).to_string(float_format="%.3f"))

beta.abs().sort_values().plot.barh()
plt.xlabel("|β|  (std. units)")
plt.title("Relative importance for log-CO₂")
plt.tight_layout()
plt.show()

# ── 7. Diagnostic checks: Breusch-Pagan and Residuals vs. Fitted ────────────────
# Compute fitted values and residuals
#fitted = lin.predict(X)
#resid  = y - fitted
fitted = results.fittedvalues
resid = results.resid

# Breusch-Pagan test for heteroscedasticity
# Requires exogenous matrix with constant term
X_const        = sm.add_constant(X)
bp_stat, bp_p, _, _ = sms.het_breuschpagan(resid, X_const)
print(f"Breusch-Pagan test: LM statistic = {bp_stat:.3f}, p-value = {bp_p:.3f}")

# Residuals vs. Fitted plot
plt.figure(figsize=(6,4))
plt.scatter(fitted, resid, alpha=0.7)
plt.axhline(0, linestyle='--', color='black')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.show()
