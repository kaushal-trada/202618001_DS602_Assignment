# 📊 Statistical Analysis & Health Insurance Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

> An end-to-end interactive data dashboard built using **Streamlit**, **Statsmodels**, and **Plotly** to translate inferential hypothesis testing, Ordinary Least Squares (OLS) regression modeling, and Gauss-Markov diagnostic checks into an accessible, responsive web application.

---

## 🚀 Live Demo

**Public Dashboard:** [Streamlit Cloud Deployment Link](https://med-insurance-dashboard.streamlit.app/)  


---

## 📌 Project Overview

This project fulfills the end-to-end statistical modeling assignment by:

- Conducting in-depth exploratory data analysis (EDA) with interactive parameter controls.
- Running adaptive two-group hypothesis testing using:
  - Shapiro-Wilk normality test
  - Levene's variance test
  - Independent *t*-test or Mann-Whitney *U* test
- Evaluating multi-group differences using One-Way ANOVA.
- Testing categorical association using the Chi-Square ($\chi^2$) test.
- Fitting an OLS Multiple Linear Regression model using `statsmodels`.
- Interpreting regression parameter estimates.
- Running comprehensive Gauss-Markov assumption diagnostics.
- Providing dynamic, user-driven predictions with 95% confidence intervals.
- Visualizing statistical results through an interactive Streamlit dashboard.

---

## 📂 Dataset Summary

- **Selected Dataset:** Option A — Medical Insurance Costs (`insurance.csv`)
- **Observations:** 1,338 rows
- **Columns:** 7
- **Target Variable:** `charges`
- **Target Description:** Annual medical charges billed by health insurance, measured in USD.

### Predictor Features

| Feature | Description |
|---|---|
| `age` | Age of primary beneficiary |
| `sex` | Insurance contractor gender (`male`, `female`) |
| `bmi` | Body Mass Index ($kg/m^2$) |
| `children` | Number of dependents covered by health insurance |
| `smoker` | Smoking status (`yes`, `no`) |
| `region` | Residential region in the US |

---

## 🔬 Statistical Findings & Key Insights

### 1. Two-Sample Hypothesis Testing — Smokers vs. Non-Smokers

The first hypothesis test investigates whether medical charges differ between smokers and non-smokers.

#### Hypotheses

- $H_0$: There is no difference in medical charges between smokers and non-smokers.

$$
H_0: \mu_{\text{smoker}} = \mu_{\text{non-smoker}}
$$

- $H_1$: There is a statistically significant difference in medical charges.

$$
H_1: \mu_{\text{smoker}} \neq \mu_{\text{non-smoker}}
$$

#### Assumption Checks

- **Shapiro-Wilk Test:**  
  Yielded $p < 0.001$ for both groups. The charges distribution exhibits substantial positive skew, violating the normality assumption.

- **Levene's Test:**  
  Yielded $p < 0.001$, indicating significant heteroscedasticity between the groups.

#### Test Selected

Because the normality assumption was violated, the **Mann-Whitney $U$ Test** was selected as the non-parametric alternative.

#### Result & Conclusion

The Mann-Whitney $U$ test produced a statistically significant result with:

$$
p < 0.001
$$

Therefore, we **reject $H_0$ at $\alpha = 0.05$**.

**Conclusion:** Smoking status has a statistically significant and substantial association with annual medical expenditures.

---

### 2. Multi-Group & Categorical Testing

#### Categorical Association — Chi-Square Test

The Chi-Square test was used to investigate the relationship between:

- `smoker`
- `region`

The test yielded:

$$
p > 0.05
$$

Therefore, we **fail to reject $H_0$**.

**Conclusion:** There is insufficient statistical evidence of an association between smoking status and geographic region in this dataset.

#### One-Way ANOVA — Charges Across Regions

One-Way ANOVA was used to evaluate whether average medical charges differ across the four regions.

The resulting test produced approximately:

$$
p \approx 0.03
$$

Since $p < 0.05$, the result is statistically significant at $\alpha = 0.05$.

**Conclusion:** Average medical charges differ significantly across regions. However, the observed regional differences may be influenced by differences in other variables, particularly BMI and smoking status, rather than geographic location alone.

---

## 3. Multiple Linear Regression Model

An Ordinary Least Squares (OLS) Multiple Linear Regression model was fitted to predict annual medical charges.

### Model Equation

The regression model can be represented as:

$$
\begin{aligned}
\text{charges} = \beta_0
&+ \beta_1(\text{age})
+ \beta_2(\text{bmi}) \\
&+ \beta_3(\text{children})
+ \beta_4(\text{smoker\_yes})
+ \varepsilon
\end{aligned}
$$

Where:

- $\beta_0$ = Intercept
- $\beta_1$ = Effect of age
- $\beta_2$ = Effect of BMI
- $\beta_3$ = Effect of number of children
- $\beta_4$ = Effect of smoking status
- $\varepsilon$ = Error term

### Model Performance

- **$R^2 \approx 0.75$**
- **Adjusted $R^2 \approx 0.749$**
- **F-statistic $p$-value < 0.001**

The model explains approximately **75% of the variation in medical charges**, indicating strong overall predictive utility.

### Important Coefficients

#### Smoking Status

$$
\beta_{\text{smoker\_yes}} \approx +23,848
$$

with:

$$
p < 0.001
$$

Holding the other variables constant, smokers are associated with approximately **$23,800 higher annual medical charges** compared with non-smokers.

#### Age

$$
\beta_{\text{age}} \approx +256
$$

with:

$$
p < 0.001
$$

Holding other variables constant, each additional year of age is associated with an estimated **$256 increase in annual medical charges**.

#### BMI

$$
\beta_{\text{bmi}} \approx +339
$$

with:

$$
p < 0.001
$$

Holding other variables constant, each one-unit increase in BMI is associated with an estimated **$339 increase in annual medical charges**.

---

## 4. Gauss-Markov Residual Diagnostics

Several diagnostic procedures were performed to evaluate the assumptions of the OLS regression model.

### Linearity & Homoscedasticity

The **Residuals vs. Fitted Values** plot was used to examine:

- Linearity
- Constant error variance
- Potential model misspecification

The plot shows some non-linear clustering and a slight heteroscedastic fan-shaped spread, suggesting that additional relationships or interactions may exist, particularly involving BMI and smoking status.

### Normality of Residuals

A **Q-Q Plot** and **Jarque-Bera Test** were used to evaluate residual normality.

The Jarque-Bera test produced:

$$
p < 0.001
$$

Therefore, the null hypothesis of normally distributed residuals is rejected.

The deviation is primarily associated with positive-tail observations and unusually high medical charges.

### Multicollinearity — VIF

Variance Inflation Factor (VIF) was calculated for the predictor variables.

The continuous predictors:

- `age`
- `bmi`
- `children`

had:

$$
\text{VIF} < 1.5
$$

This is substantially below the commonly used threshold of 5.

**Conclusion:** There is no evidence of problematic multicollinearity among the continuous predictors.

---

## 🖥️ Dashboard Architecture

The Streamlit application is divided into three functional tabs.

| Tab | Feature Set | Primary Tools |
|---|---|---|
| **Tab 1: Data Exploration** | Interactive sidebar filters, age/BMI/region controls, dynamic metrics, distribution histograms, KDE plots, and correlation heatmaps | `streamlit`, `plotly.express`, `pandas` |
| **Tab 2: Hypothesis Testing Lab** | Two-group comparison with Shapiro-Wilk, Levene's test, automatic *t*-test/Mann-Whitney selection, Chi-Square, and One-Way ANOVA | `scipy.stats`, `statsmodels` |
| **Tab 3: Live Prediction & Diagnostics** | Input controls, predicted charges, 95% confidence intervals, residual plots, Q-Q plot, and VIF tables | `statsmodels.api`, `matplotlib`, `plotly` |

---

## 🔄 Statistical Workflow

The overall workflow of the project is:

```text
Raw Insurance Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Preprocessing
        │
        ▼
Hypothesis Testing
        │
        ├── Two-Group Testing
        │     ├── Shapiro-Wilk
        │     ├── Levene
        │     └── t-test / Mann-Whitney U
        │
        ├── Chi-Square Test
        │
        └── One-Way ANOVA
        │
        ▼
Multiple Linear Regression
        │
        ▼
Model Diagnostics
        │
        ├── Residuals vs Fitted
        ├── Q-Q Plot
        ├── Jarque-Bera
        └── VIF
        │
        ▼
Live Prediction
        │
        ▼
95% Confidence Interval
```

---

## 🛠️ Installation & Local Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/kaushal-trada/202618001_DS602_Assignment.git
cd 202618001_DS602_Assignment/202618001_Lab04
```

### Step 2 — Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The application will automatically open in your browser.

Default local address:

```text
http://localhost:8501
```

---

## 📁 Repository Structure

```text
202618001_Lab04/
│
├── data/
│   └── insurance.csv
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File / Folder | Purpose |
|---|---|
| `data/insurance.csv` | Medical insurance dataset |
| `.streamlit/config.toml` | Optional Streamlit configuration |
| `app.py` | Main Streamlit dashboard |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Files excluded from Git |

---

## 📦 Dependencies

The project uses the following Python libraries:

```text
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
statsmodels>=0.14.0
plotly>=5.17.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📊 Key Technologies

### Python

Used as the primary programming language for data processing, statistical analysis, and application development.

### Pandas

Used for:

- Data loading
- Data cleaning
- Data manipulation
- Descriptive statistics

### SciPy

Used for:

- Shapiro-Wilk test
- Levene's test
- Mann-Whitney $U$ test
- Independent *t*-test
- Chi-Square test
- ANOVA

### Statsmodels

Used for:

- OLS regression
- Regression summaries
- Confidence intervals
- Statistical diagnostics
- VIF calculations

### Plotly

Used for interactive:

- Histograms
- Distribution plots
- Correlation visualizations
- Dashboard charts

### Matplotlib

Used for statistical diagnostic visualizations such as:

- Q-Q plots
- Residual plots

### Streamlit

Used to convert the statistical analysis into an interactive web dashboard.

---

## 🎯 Project Objectives

The primary objectives of this project are:

1. Perform exploratory data analysis on medical insurance data.
2. Apply appropriate statistical hypothesis tests.
3. Build a Multiple Linear Regression model.
4. Interpret regression coefficients and statistical significance.
5. Evaluate important OLS assumptions.
6. Detect multicollinearity using VIF.
7. Visualize model diagnostics.
8. Generate interactive medical charge predictions.
9. Provide 95% confidence intervals for predictions.
10. Present the complete statistical workflow through an interactive dashboard.

---

## 💡 Key Takeaways

- **Smoking status** is one of the strongest predictors of medical charges.
- **Age** has a statistically significant positive relationship with medical charges.
- **BMI** also has a statistically significant positive relationship with charges.
- The regression model explains approximately **75% of the variance** in medical charges.
- Regional differences in charges are statistically significant in the ANOVA analysis.
- Smoking status and region do not show a statistically significant categorical association.
- The residual diagnostics indicate departures from ideal normality and constant variance assumptions.
- VIF values indicate **no problematic multicollinearity** among the continuous predictors.

---

## 👤 Author

**Name:** Kaushal Trada  
**ID:** 202618001

**Course:** Statistical Analysis for Data Science  
**GitHub:** [@kaushal-trada](https://github.com/kaushal-trada)

---

## 📄 License

This project was developed for academic and educational purposes as part of the **Statistical Analysis for Data Science** coursework.