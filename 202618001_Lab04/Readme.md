# 📊 Statistical Analysis & Health Insurance Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)


> An end-to-end interactive data dashboard built using **Streamlit**, **Statsmodels**, and **Plotly** to translate inferential hypothesis testing, Ordinary Least Squares (OLS) regression modeling, and Gauss-Markov diagnostic checks into an accessible, responsive web application.

---

## 🚀 Live Demo
**Public Dashboard:** [Streamlit Cloud Deployment Link](https://share.streamlit.io/) *(Update with your deployed app URL)*

---

## 📌 Project Overview
This project fulfills the end-to-end statistical modeling assignment by:
- Conducting in-depth exploratory data analysis (EDA) with interactive parameter controls.
- Running adaptive two-group hypothesis testing (normality check via Shapiro-Wilk, variance check via Levene's test, branching into Independent $t$-test or Mann-Whitney $U$).
- Evaluating multi-group differences using One-Way ANOVA and categorical association using Chi-Square ($\chi^2$) tests.
- Fitting an OLS Multiple Linear Regression model with `statsmodels`, interpreting parameter estimates, and running comprehensive Gauss-Markov assumption diagnostics (Residuals vs. Fitted, Q-Q plots, Jarque-Bera tests, and VIF).
- Serving dynamic, user-driven predictions with 95% confidence intervals and diagnostic visualizations via an interactive UI.

---

## 📂 Dataset Summary
* **Selected Dataset:** Option A — Medical Insurance Costs (`insurance.csv`)
* **Observations:** 1,338 rows, 7 columns
* **Target Variable:** `charges` (Annual medical charges billed by health insurance, continuous in USD)
* **Predictor Features:**
  * `age`: Age of primary beneficiary (integer)
  * `sex`: Insurance contractor gender (`male`, `female`)
  * `bmi`: Body Mass Index ($kg/m^2$), objective index of body weight (continuous)
  * `children`: Number of dependents covered by health insurance (integer)
  * `smoker`: Smoking status (`yes`, `no`)
  * `region`: Beneficiary's residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`)

---

## 🔬 Statistical Findings & Key Insights

### 1. Two-Sample Hypothesis Testing (Smokers vs. Non-Smokers on Charges)
* **Hypotheses:**
  * $H_0$: There is no difference in medical charges between smokers and non-smokers ($\mu_{\text{smoker}} = \mu_{\text{non-smoker}}$).
  * $H_1$: There is a statistically significant difference in medical charges ($\mu_{\text{smoker}} \neq \mu_{\text{non-smoker}}$).
* **Assumption Checks:**
  * **Shapiro-Wilk Test:** Yielded $p < 0.001$ for both groups; the charges distribution exhibits substantial positive skew, violating the normality assumption.
  * **Levene's Test:** Yielded $p < 0.001$, indicating significant heteroscedasticity across groups.
* **Test Selected:** **Mann-Whitney $U$ Test** (non-parametric rank-sum test).
* **Result & Conclusion:** $U$-statistic was statistically significant at $p < 0.001$. We **Reject $H_0$** at $\alpha = 0.05$. Smoking status has a profound, statistically significant effect on annual medical expenditures.

### 2. Multi-Group / Categorical Testing
* **Categorical Association (Chi-Square Test):** Tested independence between `smoker` and `region`. The test yielded $p > 0.05$, **Failing to Reject $H_0$**; smoking distribution is independent of geographic region.
* **One-Way ANOVA (Charges across 4 Regions):** Evaluated whether average charges vary across regions. The resulting $F$-test yielded $p \approx 0.03$. While statistically significant at $\alpha = 0.05$, differences are driven primarily by higher average BMI in the Southeast rather than location alone.

### 3. Multiple Linear Regression Model
* **Model Equation:**
  $$\text{charges} = \beta_0 + \beta_1(\text{age}) + \beta_2(\text{bmi}) + \beta_3(\text{children}) + \beta_4(\text{smoker\_yes}) + \varepsilon$$
* **Model Performance:**
  * $R^2 \approx 0.75$, Adjusted $R^2 \approx 0.749$: The linear model accounts for approximately 75% of the total variance in medical charges.
  * $F$-statistic $p$-value $< 0.001$, confirming strong collective predictive utility.
* **Coefficients & Interpretations:**
  * $\beta_{\text{smoker\_yes}} \approx +23,848$ ($p < 0.001$): Holding all other variables constant, smoking increases expected charges by over \$23,800/year.
  * $\beta_{\text{age}} \approx +256$ ($p < 0.001$): Each additional year of age corresponds to an expected increase of \$256 in annual medical charges.
  * $\beta_{\text{bmi}} \approx +339$ ($p < 0.001$): Each 1-unit increase in BMI yields an expected \$339 increase in charges.

### 4. Gauss-Markov Residual Diagnostics
* **Linearity & Homoscedasticity (Residuals vs. Fitted):** The plot reveals non-linear clustering and slight heteroscedastic fan spread, reflecting an underlying interaction between high BMI and smoking status.
* **Normality of Residuals (Q-Q Plot & Jarque-Bera):** The Jarque-Bera test yielded $p < 0.001$, rejecting normality of residuals due to positive tail outliers (unusually heavy single-incident hospitalizations).
* **Multicollinearity (VIF):** All continuous predictor variables (`age`, `bmi`, `children`) exhibited $\text{VIF} < 1.5$, well below the critical threshold of $5.0$, confirming the absence of problematic collinearity.

---

## 🖥️ Dashboard Architecture (`app.py`)

The Streamlit web application is divided into three functional tabs:

| Tab | Feature Set | Primary Tools Used |
| :--- | :--- | :--- |
| **Tab 1: Data Exploration** | Interactive sidebar filters (age, BMI, region), dynamic metrics, distribution histograms, KDE plots, and interactive correlation heatmaps. | `streamlit`, `plotly.express`, `pandas` |
| **Tab 2: Hypothesis Testing Lab** | Dynamic selector for 2-group comparisons (automated Shapiro-Wilk, Levene's, and auto-branching $t$-test / Mann-Whitney $U$) plus Chi-Square and One-Way ANOVA tests with rendered $p$-values and conclusions. | `scipy.stats`, `statsmodels` |
| **Tab 3: Live Prediction & Diagnostics** | Input parameter controls (sliders, selectors) generating point estimates and 95% confidence intervals, paired with live residual diagnostic plots (Residuals vs. Fitted, Q-Q plot, VIF tables). | `statsmodels.api`, `matplotlib`, `plotly` |

---

## 🛠️ Installation & Local Setup

### Prerequisites
* Python 3.9+ installed
* Git

### Step-by-Step Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
   cd <your-repo-name>

   Create and activate a virtual environment:

    macOS / Linux:

    Bash
    python3 -m venv venv
    source venv/bin/activate

    Windows:

    Bash
    python -m venv venv
    venv\Scripts\activate 

    Install dependencies:

    Bash
    pip install --upgrade pip
    pip install -r requirements.txt

    Launch the Streamlit dashboard:

    Bash
    streamlit run app.py
    The application will automatically launch in your browser at http://localhost:8501.


 
## 📁 Repository Structure

```plaintext
├── data/
│   └── insurance.csv          # Medical insurance tabular dataset
├── .streamlit/
│   └── config.toml            # Optional Streamlit visual theme configuration
├── app.py                     # Primary Streamlit application source code
├── requirements.txt           # Environment dependencies
├── README.md                  # Comprehensive project documentation
└── .gitignore                 # Standard Python/IDE ignore entries


👤 Author
Name: [Your Name]

Course: Statistical Analysis for Data Science

GitHub: @your-username