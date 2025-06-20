import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from sklearn.inspection import permutation_importance

# ── 1. Load data ────────────────────────────────────────────────────────────────
CSV = Path("sensitivity_pfi_gpt_row11.csv")      # adjust if stored elsewhere
df  = pd.read_csv(CSV)

# Ensure numeric (just in case localisation inserted commas)
cols_numeric = ["parameter # (B)", "token   # (B)", "lg_alpha",
                "carbon_pred_t", "error_rate", "error_% "]
df[cols_numeric] = df[cols_numeric].apply(pd.to_numeric, errors="coerce")

# Drop any rows with NaNs that could sneak in
df = df.dropna(subset=cols_numeric).reset_index(drop=True)


# ── 2. Log-transform response and predictors (power-law scaling – Strubell ’19) ─
df["log_error_% "]   = np.log(df["error_% "]+100)
df["log_param"] = np.log(df["parameter # (B)"])
df["log_token"] = np.log(df["token   # (B)"])
df["log_eff"]   = df["lg_alpha"]

# ── 3. Standardise predictors so coefficients are comparable (Saltelli ’08) ────
X_raw = df[["token   # (B)", "parameter # (B)", "lg_alpha"]]
X     = (X_raw - X_raw.mean()) / X_raw.std(ddof=0)   # z-scores
y     = df["log_error_% "]

X_std = sm.add_constant(X)

# ── 4. Fit linear regression (OLS) ──────────────────────────────────────────────
model   = LinearRegression().fit(X_std, y)
beta  = pd.Series(model.coef_, index=X_std.columns)
lin2 = sm.OLS(y, X_std)
ols = lin2.fit()
#print(results.summary())
#print(results.summary().as_latex())
ols_robust = lin2.fit(cov_type='HC3')
#pythoprint(ols_robust.summary())
print(ols_robust.summary().as_latex())

result = permutation_importance(model, X_std, y, n_repeats=30, random_state=42)

print("Permutation Feature Importance (higher = more impact on % error):")
for name, mean_imp in zip(X_std.columns, result.importances_mean):
    print(f"  {name:4s}  {mean_imp:8.3f}")

# 6. Diagnostics -------------------------------------------------------------
#    6‑a VIF
vif = pd.DataFrame({
    "variable": X_std.columns[1:],
    "VIF": [variance_inflation_factor(X_std.values, i+1) for i in range(len(X_std.columns)-1)]
})

#    6‑b Heteroscedasticity
bp_labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
bp = dict(zip(bp_labels, het_breuschpagan(ols.resid, ols.model.exog)))

# 7. Plots -------------------------------------------------------------------
fig1 = plt.figure()
plt.scatter(ols.fittedvalues, ols.resid)
plt.axhline(0, linewidth=1)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")

fig2 = plt.figure()
sm.qqplot(ols.resid, line='45')
plt.title("Normal Q‑Q Plot")

plt.show()

# 8. Output ------------------------------------------------------------------
print("\nBreusch‑Pagan test for heteroscedasticity:", bp)
print(vif)
 