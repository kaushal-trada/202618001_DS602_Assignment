import pandas as pd
import plotly.express as px
import scipy.stats as stats


def test_smokers_vs_nonsmokers(df: pd.DataFrame) -> dict:
    """Compares charges between smokers and non-smokers."""
    smokers = df[df['smoker'] == 'yes']['charges'].dropna()
    non_smokers = df[df['smoker'] == 'no']['charges'].dropna()

    if len(smokers) == 0 or len(non_smokers) == 0:
        return {"error": "Make sure both smokers and non-smokers are included in your sidebar filters."}

    stat, p_val = stats.mannwhitneyu(smokers, non_smokers, alternative='two-sided')
    smoker_avg = smokers.mean()
    non_smoker_avg = non_smokers.mean()
    diff = smoker_avg - non_smoker_avg
    is_sig = p_val < 0.05

    return {
        "smoker_avg": smoker_avg,
        "non_smoker_avg": non_smoker_avg,
        "diff": diff,
        "p_value": p_val,
        "is_significant": is_sig,
        "verdict": f"Significant difference: Smokers pay on average ${diff:,.2f} more than non-smokers." if is_sig else "No statistically significant difference found."
    }


def test_selected_regions(df: pd.DataFrame, selected_regions: list) -> dict:
    """
    Compares charges across user-selected regions.
    Returns test statistics, group summary table, and a comparison box plot.
    """
    if len(selected_regions) < 2:
        return {"error": "Please select at least 2 regions to compare."}

    region_data = df[df['region'].isin(selected_regions)].dropna(subset=['charges'])
    groups = [sub['charges'].values for _, sub in region_data.groupby('region')]

    # One-Way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)
    is_sig = p_val < 0.05

    # Summary table for selected regions
    summary = region_data.groupby('region')['charges'].agg(
        Count='count',
        Average='mean',
        Median='median',
        Std_Dev='std'
    ).reset_index()

    summary['Average'] = summary['Average'].map('${:,.2f}'.format)
    summary['Median'] = summary['Median'].map('${:,.2f}'.format)
    summary['Std_Dev'] = summary['Std_Dev'].map('${:,.2f}'.format)

    # Comparison box plot so users can visually inspect the spread
    fig = px.box(
        region_data,
        x='region',
        y='charges',
        color='region',
        points='all',
        title="Comparison of Medical Charges Across Selected Regions"
    )
    fig.update_layout(
        template="simple_white",
        xaxis_title="Region",
        yaxis_title="Medical Charges ($)",
        showlegend=False
    )

    return {
        "p_value": p_val,
        "f_stat": f_stat,
        "is_significant": is_sig,
        "summary_table": summary,
        "box_fig": fig,
        "verdict": (
            "Statistically Significant Difference: Costs vary meaningfully between at least two of the selected regions."
            if is_sig else
            "No Significant Difference: Medical costs across these chosen regions are comparable."
        )
    }