import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Charges Dashboard", layout="wide")

df = pd.read_csv("data/insurance.csv")
import statsmodels.formula.api as smf

@st.cache_resource
def fit_model(data):
    formula = "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + bmi:C(smoker)"
    return smf.ols(formula=formula, data=data).fit()

model = fit_model(df)

st.title("Medical Insurance Charges Dashboard")

tab1, tab2, tab3 = st.tabs(["Data Exploration", "Hypothesis Testing Lab", "Live Prediction & Diagnostics"])

with tab1:
    st.header("Data Exploration")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_range = st.slider("Age range", int(df.age.min()), int(df.age.max()),
                               (int(df.age.min()), int(df.age.max())))
    with col2:
        bmi_range = st.slider("BMI range", float(df.bmi.min()), float(df.bmi.max()),
                               (float(df.bmi.min()), float(df.bmi.max())))
    with col3:
        regions = st.multiselect("Region(s)", sorted(df.region.unique()),
                                  default=sorted(df.region.unique()))

    filtered = df[
        (df.age.between(*age_range)) &
        (df.bmi.between(*bmi_range)) &
        (df.region.isin(regions))
    ]

    st.write(f"{len(filtered)} of {len(df)} records match your filters.")
    st.dataframe(filtered.head(20))
    st.subheader("Summary Statistics")
    st.dataframe(filtered[["age", "bmi", "children", "charges"]].describe())

    import plotly.express as px

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(filtered, x="charges", color="smoker", nbins=40,
                                 title="Distribution of Charges by Smoking Status")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_scatter = px.scatter(filtered, x="bmi", y="charges", color="smoker",
                                  title="BMI vs Charges")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.subheader("Correlation Matrix")
    corr = filtered[["age", "bmi", "children", "charges"]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, title="Correlation between numeric features")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.header("Hypothesis Testing Lab")

    from scipy import stats

    cat_cols = ["sex", "smoker", "region"]
    num_cols = ["age", "bmi", "children", "charges"]

    colA, colB = st.columns(2)
    with colA:
        cat_choice = st.selectbox("Categorical factor", cat_cols)
    with colB:
        num_choice = st.selectbox("Numeric metric", num_cols)

    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)

    groups_unique = df[cat_choice].unique()
    st.write(f"Groups found in {cat_choice}: {list(groups_unique)}")
    if len(groups_unique) == 2:
        st.subheader(f"Two-Group Comparison: {num_choice} by {cat_choice}")
        g1_name, g2_name = groups_unique
        g1 = df.loc[df[cat_choice] == g1_name, num_choice]
        g2 = df.loc[df[cat_choice] == g2_name, num_choice]

        sw1 = stats.shapiro(g1)
        sw2 = stats.shapiro(g2)
        lev_stat, lev_p = stats.levene(g1, g2)

        st.write(f"Shapiro-Wilk {g1_name}: p = {sw1.pvalue:.4g}")
        st.write(f"Shapiro-Wilk {g2_name}: p = {sw2.pvalue:.4g}")
        st.write(f"Levene's test p = {lev_p:.4g}")

        normal_enough = (sw1.pvalue > alpha) and (sw2.pvalue > alpha)
        if normal_enough:
            stat_val, p_val = stats.ttest_ind(g1, g2, equal_var=(lev_p > alpha))
            test_name = "Two-Sample t-test"
        else:
            stat_val, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U test"

        st.info(f"Using: {test_name}")
        st.write(f"Statistic = {stat_val:.4f}, p-value = {p_val:.4g}")

        if p_val < alpha:
            st.success(f"Reject H0 — significant difference in {num_choice} between {g1_name} and {g2_name}.")
        else:
            st.warning(f"Fail to reject H0 — no significant difference found.")

    else:
        st.subheader(f"One-Way ANOVA: {num_choice} across {cat_choice} groups")
        groups = [df.loc[df[cat_choice] == g, num_choice] for g in groups_unique]
        f_stat, p_val = stats.f_oneway(*groups)

        st.write(f"F-statistic = {f_stat:.4f}, p-value = {p_val:.4g}")

        if p_val < alpha:
            st.success(f"Reject H0 — at least one group's mean {num_choice} differs across {cat_choice}.")
        else:
            st.warning(f"Fail to reject H0 — no significant difference across groups.") 

with tab3:
    st.header("Live Prediction & Diagnostics")
    st.subheader("Predict charges for a new individual")

    c1, c2, c3 = st.columns(3)
    with c1:
        in_age = st.number_input("Age", 18, 100, 35)
        in_sex = st.selectbox("Sex", df.sex.unique())
    with c2:
        in_bmi = st.number_input("BMI", 10.0, 60.0, 28.0, step=0.1)
        in_children = st.number_input("Children", 0, 10, 0)
    with c3:
        in_smoker = st.selectbox("Smoker", df.smoker.unique())
        in_region = st.selectbox("Region", df.region.unique())

    new_point = pd.DataFrame([{
        "age": in_age, "sex": in_sex, "bmi": in_bmi,
        "children": in_children, "smoker": in_smoker, "region": in_region,
    }])

    pred = model.get_prediction(new_point)
    pred_summary = pred.summary_frame(alpha=0.05)

    st.metric("Predicted charges", f"${pred_summary['mean'].iloc[0]:,.2f}")
    st.write(f"95% confidence interval: ${pred_summary['mean_ci_lower'].iloc[0]:,.2f} — "
             f"${pred_summary['mean_ci_upper'].iloc[0]:,.2f}")
    st.write(f"95% prediction interval: ${pred_summary['obs_ci_lower'].iloc[0]:,.2f} — "
             f"${pred_summary['obs_ci_upper'].iloc[0]:,.2f}")
    st.divider()
    st.subheader("Residual Diagnostics")

    fitted_vals = model.fittedvalues
    residuals = model.resid

    d1, d2 = st.columns(2)
    with d1:
        fig_resid = px.scatter(x=fitted_vals, y=residuals,
                                labels={"x": "Fitted values", "y": "Residuals"},
                                title="Residuals vs Fitted")
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, use_container_width=True)

    with d2:
        import statsmodels.api as sm
        qq = sm.ProbPlot(residuals)
        fig_qq = px.scatter(x=qq.theoretical_quantiles, y=qq.sample_quantiles,
                             labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
                             title="Q-Q Plot of Residuals")
        st.plotly_chart(fig_qq, use_container_width=True)

    st.subheader("Multicollinearity Check (VIF)")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = sm.add_constant(df[["age", "bmi", "children"]])
    vif_df = pd.DataFrame({
        "feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    })
    st.dataframe(vif_df[vif_df.feature != "const"], hide_index=True)         