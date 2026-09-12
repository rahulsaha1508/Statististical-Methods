import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Medical Insurance Cost Analysis",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    data = pd.read_csv("insurance.csv")
    return data


df = load_data()


# ==================================================
# BUILD OLS MODEL
# ==================================================

@st.cache_resource
def build_ols_model(data):

    model_df = data.copy()

    model_df = pd.get_dummies(
        model_df,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )

    feature_columns = [
        "age",
        "bmi",
        "children",
        "sex_male",
        "smoker_yes",
        "region_northwest",
        "region_southeast",
        "region_southwest"
    ]

    X = model_df[feature_columns]
    y = model_df["charges"]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model, X, y


ols_model, X, y = build_ols_model(df)


# ==================================================
# HEADER
# ==================================================

st.title("📊 Medical Insurance Cost Analysis")

st.write(
    """
    This dashboard analyses medical insurance charges using
    exploratory data analysis, statistical hypothesis testing,
    and Ordinary Least Squares regression.
    """
)

st.sidebar.header("Dashboard Information")

st.sidebar.write(
    """
    **Dataset:** Medical Insurance Costs

    **Rows:** 1338

    **Target variable:** charges

    **Model:** Ordinary Least Squares Regression
    """
)


# ==================================================
# CREATE TABS
# ==================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Exploration",
        "🧪 Hypothesis Testing",
        "🔮 Prediction & Diagnostics"
    ]
)


# ==================================================
# TAB 1: EXPLORATION
# ==================================================

with tab1:

    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Number of Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Number of Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Average Charges",
            f"{df['charges'].mean():,.2f}"
        )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    st.subheader("Distribution of Insurance Charges")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="charges",
        kde=True,
        ax=ax1
    )

    ax1.set_title("Distribution of Medical Insurance Charges")
    ax1.set_xlabel("Insurance Charges")
    ax1.set_ylabel("Frequency")

    st.pyplot(fig1)

    st.subheader("Relationship Between BMI and Charges")

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=df,
        x="bmi",
        y="charges",
        hue="smoker",
        alpha=0.7,
        ax=ax2
    )

    ax2.set_title("BMI vs Insurance Charges")
    ax2.set_xlabel("BMI")
    ax2.set_ylabel("Charges")

    st.pyplot(fig2)

    bmi_correlation = df["bmi"].corr(df["charges"])

    st.write(
        f"**Correlation between BMI and charges:** "
        f"{bmi_correlation:.4f}"
    )

    st.subheader("Insurance Charges by Smoking Status")

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="smoker",
        y="charges",
        ax=ax3
    )

    ax3.set_title("Insurance Charges by Smoking Status")
    ax3.set_xlabel("Smoker")
    ax3.set_ylabel("Charges")

    st.pyplot(fig3)

    st.subheader("Average Charges by Region")

    region_average = (
        df.groupby("region")["charges"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(region_average)


# ==================================================
# TAB 2: HYPOTHESIS TESTING
# ==================================================

with tab2:

    st.header("Statistical Hypothesis Testing")

    st.write(
        """
        Select a variable to compare insurance charges across
        different groups.
        """
    )

    test_choice = st.selectbox(
        "Select hypothesis test",
        [
            "Charges by Sex",
            "Charges by Smoking Status",
            "Charges by Region"
        ]
    )

    # --------------------------------------------------
    # TEST 1: SEX
    # --------------------------------------------------

    if test_choice == "Charges by Sex":

        st.subheader("Mann-Whitney U Test: Charges by Sex")

        female_charges = df[
            df["sex"] == "female"
        ]["charges"]

        male_charges = df[
            df["sex"] == "male"
        ]["charges"]

        u_statistic, p_value = stats.mannwhitneyu(
            female_charges,
            male_charges,
            alternative="two-sided"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Mann-Whitney U Statistic",
                f"{u_statistic:.4f}"
            )

        with col2:
            st.metric(
                "p-value",
                f"{p_value:.6f}"
            )

        st.write(
            f"Average charges for females: "
            f"**{female_charges.mean():,.2f}**"
        )

        st.write(
            f"Average charges for males: "
            f"**{male_charges.mean():,.2f}**"
        )

        if p_value < 0.05:
            st.error(
                "Reject H₀: There is a statistically significant "
                "difference in charges between males and females."
            )
        else:
            st.success(
                "Fail to reject H₀: There is no statistically "
                "significant difference in charges between males "
                "and females."
            )

    # --------------------------------------------------
    # TEST 2: SMOKER
    # --------------------------------------------------

    elif test_choice == "Charges by Smoking Status":

        st.subheader(
            "Mann-Whitney U Test: Charges by Smoking Status"
        )

        non_smoker_charges = df[
            df["smoker"] == "no"
        ]["charges"]

        smoker_charges = df[
            df["smoker"] == "yes"
        ]["charges"]

        u_statistic, p_value = stats.mannwhitneyu(
            non_smoker_charges,
            smoker_charges,
            alternative="two-sided"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Mann-Whitney U Statistic",
                f"{u_statistic:.4f}"
            )

        with col2:
            st.metric(
                "p-value",
                f"{p_value:.6e}"
            )

        st.write(
            f"Average charges for non-smokers: "
            f"**{non_smoker_charges.mean():,.2f}**"
        )

        st.write(
            f"Average charges for smokers: "
            f"**{smoker_charges.mean():,.2f}**"
        )

        if p_value < 0.05:
            st.error(
                "Reject H₀: Smoking status is significantly "
                "associated with insurance charges."
            )
        else:
            st.success(
                "Fail to reject H₀: No statistically significant "
                "difference was found."
            )

    # --------------------------------------------------
    # TEST 3: REGION
    # --------------------------------------------------

    elif test_choice == "Charges by Region":

        st.subheader("One-Way ANOVA: Charges by Region")

        northeast = df[
            df["region"] == "northeast"
        ]["charges"]

        northwest = df[
            df["region"] == "northwest"
        ]["charges"]

        southeast = df[
            df["region"] == "southeast"
        ]["charges"]

        southwest = df[
            df["region"] == "southwest"
        ]["charges"]

        f_statistic, p_value = stats.f_oneway(
            northeast,
            northwest,
            southeast,
            southwest
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "F-statistic",
                f"{f_statistic:.4f}"
            )

        with col2:
            st.metric(
                "p-value",
                f"{p_value:.6f}"
            )

        st.write("Average charges by region:")

        region_means = (
            df.groupby("region")["charges"]
            .mean()
            .round(2)
        )

        st.dataframe(region_means)

        if p_value < 0.05:
            st.error(
                "Reject H₀: At least one region has a "
                "statistically different mean insurance charge."
            )
        else:
            st.success(
                "Fail to reject H₀: There is no statistically "
                "significant difference among the regions."
            )


# ==================================================
# TAB 3: PREDICTION AND DIAGNOSTICS
# ==================================================

with tab3:

    st.header("OLS Prediction and Diagnostics")

    st.subheader("OLS Model Summary")

    st.write(
        f"**R-squared:** {ols_model.rsquared:.4f}"
    )

    st.write(
        f"**Adjusted R-squared:** "
        f"{ols_model.rsquared_adj:.4f}"
    )

    st.write(
        f"**F-statistic:** {ols_model.fvalue:.4f}"
    )

    st.write(
        f"**Overall model p-value:** "
        f"{ols_model.f_pvalue:.6e}"
    )

    st.subheader("Regression Coefficients")

    coefficient_table = pd.DataFrame({
        "Coefficient": ols_model.params,
        "p-value": ols_model.pvalues,
        "CI Lower": ols_model.conf_int()[0],
        "CI Upper": ols_model.conf_int()[1]
    })

    st.dataframe(
        coefficient_table.round(4),
        use_container_width=True
    )

    st.subheader("Variance Inflation Factor")

    vif_table = pd.DataFrame()

    vif_table["Feature"] = X.columns

    vif_table["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    st.dataframe(
        vif_table.round(4),
        use_container_width=True
    )

    st.subheader("Residuals vs Fitted Values")

    fitted_values = ols_model.fittedvalues
    residuals = ols_model.resid

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    ax4.scatter(
        fitted_values,
        residuals,
        alpha=0.5
    )

    ax4.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    ax4.set_title("Residuals vs Fitted Values")
    ax4.set_xlabel("Fitted Values")
    ax4.set_ylabel("Residuals")

    st.pyplot(fig4)

    st.subheader("Q-Q Plot")

    fig5 = sm.qqplot(
        residuals,
        line="45",
        fit=True
    )

    st.pyplot(fig5.figure)

    st.subheader("Jarque-Bera Test")

    jb_statistic, jb_p_value = stats.jarque_bera(
        residuals
    )

    st.write(
        f"**Jarque-Bera Statistic:** "
        f"{jb_statistic:.4f}"
    )

    st.write(
        f"**Jarque-Bera p-value:** "
        f"{jb_p_value:.6e}"
    )

    if jb_p_value < 0.05:
        st.warning(
            "Reject H₀: The residuals are not normally distributed."
        )
    else:
        st.success(
            "Fail to reject H₀: The residuals may be normally distributed."
        )

    # --------------------------------------------------
    # PREDICTION SECTION
    # --------------------------------------------------

    st.subheader("Predict Medical Insurance Charges")

    st.write(
        "Enter the details below to estimate insurance charges."
    )

    age_input = st.slider(
        "Age",
        min_value=18,
        max_value=64,
        value=30
    )

    bmi_input = st.slider(
        "BMI",
        min_value=15.0,
        max_value=55.0,
        value=25.0,
        step=0.1
    )

    children_input = st.slider(
        "Number of Children",
        min_value=0,
        max_value=5,
        value=0
    )

    sex_input = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    smoker_input = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region_input = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

    if st.button("Predict Insurance Charges"):

        prediction_data = pd.DataFrame({
            "age": [age_input],
            "bmi": [bmi_input],
            "children": [children_input],
            "sex_male": [
                1 if sex_input == "male" else 0
            ],
            "smoker_yes": [
                1 if smoker_input == "yes" else 0
            ],
            "region_northwest": [
                1 if region_input == "northwest" else 0
            ],
            "region_southeast": [
                1 if region_input == "southeast" else 0
            ],
            "region_southwest": [
                1 if region_input == "southwest" else 0
            ]
        })

        prediction_data = sm.add_constant(
            prediction_data,
            has_constant="add"
        )

        prediction_data = prediction_data[
            X.columns
        ]

        prediction_result = ols_model.get_prediction(
            prediction_data
        )

        prediction_summary = prediction_result.summary_frame(
            alpha=0.05
        )

        predicted_charge = prediction_summary[
            "mean"
        ].iloc[0]

        mean_ci_lower = prediction_summary[
            "mean_ci_lower"
        ].iloc[0]

        mean_ci_upper = prediction_summary[
            "mean_ci_upper"
        ].iloc[0]

        obs_ci_lower = prediction_summary[
            "obs_ci_lower"
        ].iloc[0]

        obs_ci_upper = prediction_summary[
            "obs_ci_upper"
        ].iloc[0]

        st.success(
            f"Estimated Insurance Charges: "
            f"₹{predicted_charge:,.2f}"
        )

        st.write(
            f"**95% Confidence Interval for Mean Prediction:** "
            f"₹{mean_ci_lower:,.2f} to ₹{mean_ci_upper:,.2f}"
        )

        st.write(
            f"**95% Prediction Interval for an Individual:** "
            f"₹{obs_ci_lower:,.2f} to ₹{obs_ci_upper:,.2f}"
        )