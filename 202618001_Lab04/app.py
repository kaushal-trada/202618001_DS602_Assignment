import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.eda import (
    get_summary_table,
    plot_simple_distribution,
    plot_correlation_heatmap,
    plot_simple_scatter
)
from src.hypothesis import test_smokers_vs_nonsmokers, test_selected_regions
from src.modeling import (
    train_insurance_model,
    predict_cost,
    plot_prediction_gauge,
    get_model_diagnostics
)

# --- DATA LOADING & PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATA_PATH = BASE_DIR / "data" / "insurance.csv"
REMOTE_DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

@st.cache_data
def load_data():
    if LOCAL_DATA_PATH.exists():
        return pd.read_csv(LOCAL_DATA_PATH)
    try:
        return pd.read_csv(REMOTE_DATA_URL)
    except Exception as err:
        st.error(f"Failed to load dataset from local path and remote fallback: {err}")
        st.stop()

df = load_data()

# --- GLOBAL SIDEBAR FILTERS ---
st.sidebar.header("Filter Dataset")
st.sidebar.caption("Filter the records used across all tabs.")

age_min, age_max = int(df['age'].min()), int(df['age'].max())
selected_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

bmi_min, bmi_max = float(df['bmi'].min()), float(df['bmi'].max())
selected_bmi = st.sidebar.slider("BMI Range", bmi_min, bmi_max, (bmi_min, bmi_max))

all_smokers = df['smoker'].unique().tolist()
selected_smokers = st.sidebar.multiselect("Smoker Status", options=all_smokers, default=all_smokers)

all_regions = df['region'].unique().tolist()
selected_regions = st.sidebar.multiselect("Regions", options=all_regions, default=all_regions)

filtered_df = df[
    (df['age'].between(selected_age[0], selected_age[1])) &
    (df['bmi'].between(selected_bmi[0], selected_bmi[1])) &
    (df['smoker'].isin(selected_smokers)) &
    (df['region'].isin(selected_regions))
]

st.sidebar.info(f"Showing **{len(filtered_df):,}** of **{len(df):,}** people ({len(filtered_df)/len(df)*100:.1f}%)")

# --- MAIN INTERFACE ---
st.title("Medical Insurance Cost & Statistics Lab")
st.write("Explore demographic trends, run statistical tests, and forecast individual medical costs.")

tab1, tab2, tab3 = st.tabs([
    "1. Data Exploration (EDA)",
    "2. Hypothesis Testing",
    "3. Cost Estimator & Model Health"
])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tab1:
    st.subheader("1. General Summary Statistics")
    st.dataframe(get_summary_table(filtered_df), use_container_width=True)
    st.caption(
        f"**Summary Takeaway:** The average medical charge for this subset is "
        f"**${filtered_df['charges'].mean():,.2f}**, while the typical middle individual (median) pays "
        f"**${filtered_df['charges'].median():,.2f}**. A large gap between mean and median indicates that high medical outliers pull up the overall average."
    )

    st.markdown("---")
    st.subheader("2. Visual Distribution & Relationships")
    col1, col2 = st.columns(2)

    with col1:
        dist_var = st.selectbox("Feature to Inspect", ["charges", "bmi", "age"], index=0)
        group_var = st.selectbox("Split / Color by", [None, "smoker", "sex", "region"], index=1)
        st.plotly_chart(plot_simple_distribution(filtered_df, dist_var, group_var), use_container_width=True)
        st.caption(f"**Chart Insight:** Shows the spread of **{dist_var}**. The horizontal box on top highlights median and outlier boundaries.")

    with col2:
        scatter_x = st.selectbox("Compare with Charges (X-axis)", ["age", "bmi", "children"], index=0)
        st.plotly_chart(plot_simple_scatter(filtered_df, scatter_x, "charges", group_var), use_container_width=True)
        st.caption(f"**Chart Insight:** Upward-sloping trendlines suggest an increase in **{scatter_x}** directly associates with higher medical bills.")

    st.markdown("---")
    st.subheader("3. Feature Correlation Matrix")
    st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True)
    st.caption(
        "**How to read this:** Values near 1.00 signify a strong positive connection (as one increases, the other does too). "
        "Values near 0.00 indicate negligible linear correlation."
    )

# ==========================================
# TAB 2: HYPOTHESIS TESTING
# ==========================================
with tab2:
    st.subheader("Hypothesis Test 1: Does Smoking Increase Medical Costs?")
    st.write("We compare average costs between non-smokers and smokers using the active filtered data.")
    
    t1_res = test_smokers_vs_nonsmokers(filtered_df)
    if "error" in t1_res:
        st.warning(t1_res["error"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Smoker Average", f"${t1_res['smoker_avg']:,.2f}")
        c2.metric("Non-Smoker Average", f"${t1_res['non_smoker_avg']:,.2f}")
        c3.metric("Cost Gap", f"+${t1_res['diff']:,.2f}")

        if t1_res['is_significant']:
            st.success(f"**Statistical Verdict:** {t1_res['verdict']}")
        else:
            st.info(f"**Statistical Verdict:** {t1_res['verdict']}")

        st.caption(
            f"**Explanation:** The test generated a p-value of `{t1_res['p_value']:.4e}`. "
            "Because this is less than the standard 0.05 threshold, the difference is statistically significant and not due to chance."
        )

    st.markdown("---")
    st.subheader("Hypothesis Test 2: Does Region Influence Medical Costs?")
    st.write("Pick two or more regions to verify whether location plays a significant role in insurance expenses.")

    chosen_regions = st.multiselect(
        "Select regions to compare:",
        options=all_regions,
        default=all_regions
    )

    t2_res = test_selected_regions(filtered_df, chosen_regions)
    if "error" in t2_res:
        st.warning(t2_res["error"])
    else:
        r_left, r_right = st.columns([1, 1])

        with r_left:
            st.markdown("##### Regional Cost Summary")
            st.dataframe(t2_res['summary_table'], use_container_width=True)

            if t2_res['is_significant']:
                st.success(f"**Test Result:** {t2_res['verdict']}")
            else:
                st.info(f"**Test Result:** {t2_res['verdict']}")

            st.caption(
                f"**Explanation:** The ANOVA test yielded `F = {t2_res['f_stat']:.3f}` and `p = {t2_res['p_value']:.4f}`. "
                + ("Because p < 0.05, regional disparities are meaningful." if t2_res['is_significant'] else "Because p >= 0.05, observed variations between these regions are within normal random range.")
            )

        with r_right:
            st.plotly_chart(t2_res['box_fig'], use_container_width=True)
            st.caption("**Visual Proof:** Inspect the box heights and medians across each region to see where the spread differs.")

# ==========================================
# TAB 3: PREDICTIONS & DIAGNOSTICS
# ==========================================
with tab3:
    st.subheader("Live Insurance Cost Estimator")

    # Fit regression model
    model, feature_cols = train_insurance_model(df)

    col_form, col_pred = st.columns([1, 1])

    with col_form:
        st.markdown("#### Step 1: Provide Individual Characteristics")
        st.caption("Adjust the parameters to represent the individual you want to price.")

        p_age = st.slider("Age", 18, 65, 30)
        p_bmi = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 27.5, step=0.1)
        p_children = st.selectbox("Dependent Children", [0, 1, 2, 3, 4, 5], index=0)
        p_sex = st.radio("Biological Sex", ["male", "female"], horizontal=True)
        p_smoker = st.radio("Smoking Status", ["no", "yes"], horizontal=True)
        p_region = st.selectbox("Residential Region", all_regions)

        user_input = {
            "age": p_age, "bmi": p_bmi, "children": p_children,
            "sex": p_sex, "smoker": p_smoker, "region": p_region
        }

    with col_pred:
        st.markdown("#### Step 2: Estimated Billing Output")
        pred_dict = predict_cost(model, feature_cols, user_input)

        st.plotly_chart(
            plot_prediction_gauge(pred_dict['predicted'], pred_dict['likely_min'], pred_dict['likely_max']),
            use_container_width=True
        )

        st.markdown(f"### Expected Bill: **${pred_dict['predicted']:,.2f}**")
        st.info(f"**Likely Individual Range:** between **${pred_dict['likely_min']:,.2f}** and **${pred_dict['likely_max']:,.2f}**")

        st.markdown("""
        **What does this output signify?**
        * **Expected Bill:** The average cost for an individual with these exact traits.
        * **Likely Range:** A 95% certainty interval accounting for personal health variances, unplanned treatments, and emergency care.
        * **Major Cost Drivers:** In this dataset, smoking status and high BMI carry the highest cost penalties.
        """)

    st.markdown("---")
    st.subheader("Model Diagnostic & Reliability Check")
    fig_diag, r2_pct = get_model_diagnostics(model, df)

    diag_left, diag_right = st.columns([1, 1])
    with diag_left:
        st.plotly_chart(fig_diag, use_container_width=True)

    with diag_right:
        st.markdown(f"#### Model Accuracy: **{r2_pct}%**")
        st.write(
            f"The regression model accounts for **{r2_pct}%** of all price variation in the insurance dataset."
        )
        st.caption(
            "**How to interpret this diagnostic chart:**\n"
            "- Each dot represents one person in the dataset.\n"
            "- The **red dashed line** indicates a perfect prediction.\n"
            "- Dots clustered close to the red line represent accurate predictions. Notice the distinct secondary tier above $30,000, primarily corresponding to smokers."
        )