# Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

**Course:** Statistical Modeling with Python (M.Sc. Data Science, Sem 1)
**Dataset:** Medical Insurance Costs (`data/insurance.csv`, 1,338 rows)

## How to run

```bash
python -m venv venv
venv\Scripts\activate      # Windows; use source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
streamlit run app.py       # opens at http://localhost:8501
```

## Dataset Summary

| Column | Type | Description |
|---|---|---|
| age | numeric | Age of primary beneficiary |
| sex | categorical | male / female |
| bmi | numeric | Body mass index |
| children | numeric | Number of dependents |
| smoker | categorical | yes / no |
| region | categorical | northeast / northwest / southeast / southwest |
| charges | numeric | Individual medical costs billed (target variable) |

No missing values in any column. `charges` is heavily right-skewed — driven almost
entirely by the smoker subgroup (274 of 1,338 people), whose average charges
(~$32,050) are roughly 4x higher than non-smokers (~$8,434) and whose max charge
(~$63,770) is far beyond the non-smoker max (~$36,910).

## Statistical Findings

### Hypothesis Test 1 — Smoker vs Non-Smoker Charges (Two-Group Comparison)

- H0: Mean charges are equal for smokers and non-smokers. H1: They differ.
- Shapiro-Wilk rejected normality for both groups (smokers: p = 3.62e-9;
  non-smokers: p = 1.45e-28), and Levene's test rejected equal variance
  (p = 1.56e-66) — so a **Mann-Whitney U test** was used instead of a t-test.
- Result: U = 284133, p = 5.27e-130.
- **Conclusion: Reject H0.** Smokers are charged significantly more than
  non-smokers.

### Hypothesis Test 2 — One-Way ANOVA (Charges Across 4 Regions)

- H0: Mean charges are equal across all four regions. H1: At least one differs.
- Result: F = 2.9696, p = 0.03089.
- **Conclusion: Reject H0** at α = 0.05, though the effect is far weaker than
  the smoking effect (p = 0.031 vs. p ≈ 10⁻¹³⁰) — statistically significant,
  but practically small.

### OLS Regression

Model: `charges ~ age + bmi + children + sex + smoker + region + bmi:smoker`

- **R² = 0.841, Adjusted R² = 0.840** — the model explains ~84% of the
  variation in charges.
- **Significant predictors (p < 0.001):** age (+$263.62/year), children
  (+$516.40 per child), smoker, and the bmi:smoker interaction (+$1,443.10
  per BMI point, for smokers only).
- **Not significant:** plain `bmi` (p = 0.358) and `sex` (p = 0.061).
- **Note on the smoker coefficient:** `C(smoker)[T.yes]` shows as -20,415,
  which looks like smoking *lowers* charges. This is a modeling artifact —
  because of the bmi:smoker interaction, this coefficient only represents the
  effect at BMI = 0 (not realistic). At an average BMI of ~30.66, the true
  combined effect of smoking is approximately **+$23,850**, consistent with
  the large gap found in Hypothesis Test 1.

**Gauss-Markov Diagnostics:**

- **Multicollinearity (VIF):** age = 1.014, bmi = 1.012, children = 1.002 —
  all near 1, well below the threshold of 5. **No multicollinearity problem.**
- **Homoscedasticity:** The Residuals vs Fitted plot shows a clear
  non-random pattern — variance is tighter at low fitted values and widens/
  trends downward at higher fitted values. **Heteroscedasticity is present**,
  likely because smokers and non-smokers form two distinct bands of residual
  behavior.
- **Normality of residuals:** The Q-Q plot shows several points curving sharply
  away from the 45° line at the upper tail (residuals of 5-6+ standard
  deviations). **Residuals are not normally distributed**, consistent with the
  right-skew already observed in `charges` and the smoker subgroup.
- **Overall:** the model fits well and has no multicollinearity issues, but
  violates the homoscedasticity and normality assumptions — both traceable to
  the same underlying right-skew caused by the smoker subgroup. Coefficient
  estimates remain unbiased, but standard errors may be understated; a
  log-transform of `charges` or robust standard errors would be a reasonable
  next step.

## Dashboard (`app.py`)

Three tabs:

1. **Data Exploration** — sliders for age/BMI range, a region multi-select,
   summary statistics, a charges histogram by smoking status, a BMI-vs-charges
   scatter plot, and a correlation heatmap — all reactive to the filters.
2. **Hypothesis Testing Lab** — select any categorical factor and numeric
   metric; the app automatically runs a t-test/Mann-Whitney U test (2 groups)
   or One-Way ANOVA (3+ groups) based on a Shapiro-Wilk normality check, and
   reports Reject/Fail to Reject H0 at a user-adjustable α.
3. **Live Prediction & Diagnostics** — enter a hypothetical individual's
   details to get a predicted charge with 95% confidence and prediction
   intervals, plus live residual, Q-Q, and VIF diagnostics for the fitted model.


**Live demo:** _(add link here after deployment)_
