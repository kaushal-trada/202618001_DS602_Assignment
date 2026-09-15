import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm


def train_insurance_model(df: pd.DataFrame):
    """Fits an OLS model and returns model object and feature columns."""
    clean_df = df.dropna().copy()
    features = clean_df[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
    y = clean_df['charges']

    X = pd.get_dummies(features, drop_first=True, dtype=float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model, X.columns.tolist()


def predict_cost(model, feature_columns: list, user_inputs: dict) -> dict:
    """Generates point estimate, 95% observation interval, and relative benchmark."""
    row = pd.Series(0.0, index=feature_columns)
    row['const'] = 1.0
    row['age'] = user_inputs['age']
    row['bmi'] = user_inputs['bmi']
    row['children'] = user_inputs['children']

    if f"sex_{user_inputs['sex']}" in row:
        row[f"sex_{user_inputs['sex']}"] = 1.0
    if f"smoker_{user_inputs['smoker']}" in row:
        row[f"smoker_{user_inputs['smoker']}"] = 1.0
    if f"region_{user_inputs['region']}" in row:
        row[f"region_{user_inputs['region']}"] = 1.0

    design_matrix = pd.DataFrame([row])
    pred = model.get_prediction(design_matrix).summary_frame(alpha=0.05).iloc[0]

    point_est = round(pred['mean'], 2)
    min_est = round(max(0, pred['obs_ci_lower']), 2)
    max_est = round(pred['obs_ci_upper'], 2)

    return {
        "predicted": point_est,
        "likely_min": min_est,
        "likely_max": max_est
    }


def plot_prediction_gauge(pred_val: float, min_val: float, max_val: float):
    """Displays where the estimate falls against typical costs ($1k to $50k+)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_val,
        number={'prefix': "$", 'valueformat': ",.0f"},
        title={'text': "Predicted Annual Bill"},
        gauge={
            'axis': {'range': [0, 60000], 'tickformat': "$,.0f"},
            'bar': {'color': "#1A1A1A"},
            'steps': [
                {'range': [0, 15000], 'color': "#D8F3DC"},
                {'range': [15000, 30000], 'color': "#FFE3A8"},
                {'range': [30000, 60000], 'color': "#FFC6C6"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': pred_val
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), template="simple_white")
    return fig


def get_model_diagnostics(model, df: pd.DataFrame):
    """Generates an intuitive Actual vs. Predicted scatter plot."""
    clean_df = df.dropna().copy()
    y_actual = clean_df['charges']
    y_pred = model.fittedvalues

    diag_df = pd.DataFrame({"Actual Bill": y_actual, "Predicted Bill": y_pred})
    fig = px.scatter(
        diag_df,
        x="Actual Bill",
        y="Predicted Bill",
        opacity=0.5,
        title="Actual vs. Predicted Medical Bills"
    )
    max_val = max(y_actual.max(), y_pred.max())
    fig.add_shape(
        type="line", line=dict(dash="dash", color="red"),
        x0=0, y0=0, x1=max_val, y1=max_val
    )
    fig.update_layout(
        template="simple_white",
        xaxis_title="Actual Charges ($)",
        yaxis_title="Model Predicted Charges ($)"
    )
    return fig, round(model.rsquared * 100, 1)