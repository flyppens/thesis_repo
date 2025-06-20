import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



CSV = Path("sensitivity_row11.csv")      
df  = pd.read_csv(CSV)
df.rename(columns=lambda x: x.strip(), inplace=True)          
df['y'] = np.log(df['error_%'] + 100)


X = df[['token # (B)', 'parameter # (B)', 'hardware efficiency']]
X_std = (X - X.mean()) / X.std()                              
X_std = sm.add_constant(X_std)


ols = sm.OLS(df['y'], X_std).fit()
lin = LinearRegression().fit(X_std, df['y'])

ols_robust = ols.get_robustcov_results(cov_type='HC3')

result = permutation_importance(lin, X_std, df['y'], n_repeats=30, random_state=42)

print("Permutation Feature Importance (higher = more impact on % error):")
for name, mean_imp in zip(X_std.columns, result.importances_mean):
    print(f"  {name:4s}  {mean_imp:8.3f}")


vif = pd.DataFrame({
    "variable": X_std.columns[1:],
    "VIF": [variance_inflation_factor(X_std.values, i+1) for i in range(len(X_std.columns)-1)]
})


bp_labels = ["LM Statistic", "LM p-value", "F Statistic", "F p-value"]
bp = dict(zip(bp_labels, het_breuschpagan(ols.resid, ols.model.exog)))


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
print("=== Ordinary Least Squares (with classical SEs) ===")
print(ols.summary())
print(ols.summary().as_latex())
print("\n=== Same model with HC3 robust SEs ===")
print(ols_robust.summary())
print(ols_robust.summary().as_latex())
print("\nBreusch‑Pagan test for heteroscedasticity:", bp)
print(vif)
 
