import numpy as np
import pandas as pd
import plotly.express as px


def get_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates clean descriptive metrics."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = []
    for col in num_cols:
        s = df[col].dropna()
        stats.append({
            "Feature": col,
            "Average (Mean)": round(s.mean(), 2),
            "Typical Middle (Median)": round(s.median(), 2),
            "Spread (Std Dev)": round(s.std(), 2),
            "Lowest": round(s.min(), 2),
            "Highest": round(s.max(), 2)
        })
    return pd.DataFrame(stats)


def plot_simple_distribution(df: pd.DataFrame, col: str, group_by: str = None):
    """Standard intuitive histogram with box plot on top."""
    fig = px.histogram(
        df,
        x=col,
        color=group_by,
        barmode="overlay",
        marginal="box",
        opacity=0.7,
        title=f"Distribution of {col.capitalize()}" + (f" (Grouped by {group_by})" if group_by else "")
    )
    fig.update_layout(
        template="simple_white",
        xaxis_title=col.capitalize(),
        yaxis_title="Count of Individuals"
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix with clear color scale and numbers."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Feature Correlation Matrix (Closer to 1.00 = Stronger Relationship)"
    )
    fig.update_layout(template="simple_white")
    return fig


def plot_simple_scatter(df: pd.DataFrame, x_col: str, y_col: str, group_by: str = None):
    """Clean scatter plot with an OLS trendline."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=group_by,
        trendline="ols",
        opacity=0.6,
        title=f"{x_col.capitalize()} vs. {y_col.capitalize()}"
    )
    fig.update_layout(
        template="simple_white",
        xaxis_title=x_col.capitalize(),
        yaxis_title=y_col.capitalize()
    )
    return fig