"""
Utility functions for forecasting pipeline
Includes metrics calculation, data formatting, and visualization helpers
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Absolute Percentage Error

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        float: WAPE percentage
    """
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_upper: np.ndarray = None) -> dict:
    """
    Calculate comprehensive evaluation metrics

    Args:
        y_true: Actual values
        y_pred: Predicted values (median/Q50)
        y_upper: Upper bound predictions (Q90), optional

    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    metrics = {}

    # Error metrics
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['WAPE'] = calculate_wape(y_true, y_pred)

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    metrics['R2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Calibration (if upper bound provided)
    if y_upper is not None:
        metrics['Calibration_Q90'] = 100 * np.mean(y_true <= y_upper)
        metrics['Stockout_Risk'] = 100 * np.mean(y_true > y_upper)

    return metrics


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format prediction DataFrame for display

    Args:
        df: Raw prediction DataFrame

    Returns:
        pd.DataFrame: Formatted table
    """
    # Select and rename columns
    display_columns = {
        'SKU': 'SKU',
        'product_type': 'Product Type',
        'tier': 'Tier',
        'manufacturing_time': 'Lead Time (days)',
        'prob_nonzero': 'Demand Probability',
        'pred_median': 'Forecast (Q50)',
        'pred_upper': 'Safety Stock (Q90)',
        'historical_avg_sales': 'Historical Avg'
    }

    df_display = df[list(display_columns.keys())].copy()
    df_display.columns = list(display_columns.values())

    # Format numeric columns
    df_display['Demand Probability'] = df_display['Demand Probability'].apply(lambda x: f"{x:.1%}")
    df_display['Forecast (Q50)'] = df_display['Forecast (Q50)'].apply(lambda x: f"{x:.1f}")
    df_display['Safety Stock (Q90)'] = df_display['Safety Stock (Q90)'].apply(lambda x: f"{x:.1f}")
    df_display['Historical Avg'] = df_display['Historical Avg'].apply(lambda x: f"{x:.2f}")

    return df_display


def create_visualization_data(df: pd.DataFrame) -> dict:
    """
    Prepare data for various visualizations

    Args:
        df: Prediction DataFrame

    Returns:
        dict: Dictionary with data for different charts
    """
    viz_data = {}

    # Top 10 SKUs by predicted demand
    top_skus = df.nlargest(10, 'pred_median')[['SKU', 'pred_median', 'pred_upper']]
    viz_data['top_skus'] = top_skus

    # Demand by product type
    by_product = df.groupby('product_type').agg({
        'pred_median': 'sum',
        'pred_upper': 'sum'
    }).reset_index()
    viz_data['by_product_type'] = by_product

    # Demand by tier
    by_tier = df.groupby('tier').agg({
        'SKU': 'count',
        'pred_median': 'sum',
        'pred_upper': 'sum'
    }).reset_index()
    by_tier.columns = ['tier', 'sku_count', 'pred_median', 'pred_upper']
    viz_data['by_tier'] = by_tier

    # Manufacturing time vs demand
    mfg_demand = df[['manufacturing_time', 'pred_median', 'tier']].copy()
    viz_data['mfg_vs_demand'] = mfg_demand

    return viz_data


def get_alert_skus(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Identify SKUs that need attention (high demand, high stockout risk)

    Args:
        df: Prediction DataFrame
        threshold: Probability threshold for alerts

    Returns:
        pd.DataFrame: SKUs requiring attention
    """
    alerts = df[
        (df['prob_nonzero'] >= threshold) &
        (df['pred_median'] > 0)
    ].copy()

    alerts = alerts.sort_values('pred_median', ascending=False)

    return alerts[['SKU', 'product_type', 'tier', 'manufacturing_time',
                   'prob_nonzero', 'pred_median', 'pred_upper']]


def calculate_order_recommendations(df: pd.DataFrame, current_inventory: dict = None) -> pd.DataFrame:
    """
    Calculate recommended order quantities based on forecasts

    Args:
        df: Prediction DataFrame
        current_inventory: Optional dict of {SKU: current_stock}

    Returns:
        pd.DataFrame: Order recommendations
    """
    recommendations = df[['SKU', 'manufacturing_time', 'pred_median', 'pred_upper']].copy()

    # If current inventory provided, calculate deficit
    if current_inventory:
        recommendations['current_stock'] = recommendations['SKU'].map(current_inventory).fillna(0)
        recommendations['deficit_q50'] = (recommendations['pred_median'] - recommendations['current_stock']).clip(lower=0)
        recommendations['deficit_q90'] = (recommendations['pred_upper'] - recommendations['current_stock']).clip(lower=0)
    else:
        # Assume zero inventory
        recommendations['current_stock'] = 0
        recommendations['deficit_q50'] = recommendations['pred_median']
        recommendations['deficit_q90'] = recommendations['pred_upper']

    # Add priority based on deficit and manufacturing time
    recommendations['priority_score'] = (
        recommendations['deficit_q90'] / (recommendations['manufacturing_time'] + 1)
    )

    recommendations = recommendations.sort_values('priority_score', ascending=False)

    return recommendations


def aggregate_monthly_sales(df: pd.DataFrame, column: str = 'quantity') -> pd.DataFrame:
    """
    Aggregate sales data by month (avoiding double-counting from rolling windows)

    Args:
        df: DataFrame with date and sales columns
        column: Column to aggregate

    Returns:
        pd.DataFrame: Monthly aggregated data
    """
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    monthly = df.groupby('year_month')[column].sum().reset_index()
    monthly.columns = ['year_month', 'total']

    return monthly


def create_performance_summary(df_test: pd.DataFrame) -> dict:
    """
    Create a summary of model performance for dashboard display

    Args:
        df_test: Test dataset with actual and predicted values

    Returns:
        dict: Performance summary
    """
    summary = {}

    # Overall metrics
    metrics = calculate_metrics(
        df_test['target_lead_time_demand'].values,
        df_test['pred_median'].values,
        df_test['pred_upper'].values
    )

    summary['overall'] = metrics

    # Metrics by tier
    tier_metrics = {}
    for tier in sorted(df_test['tier'].unique()):
        df_tier = df_test[df_test['tier'] == tier]
        tier_metrics[f'tier_{tier}'] = calculate_metrics(
            df_tier['target_lead_time_demand'].values,
            df_tier['pred_median'].values,
            df_tier['pred_upper'].values
        )

    summary['by_tier'] = tier_metrics

    return summary


def format_currency(value: float, currency: str = '€') -> str:
    """
    Format value as currency

    Args:
        value: Numeric value
        currency: Currency symbol

    Returns:
        str: Formatted currency string
    """
    return f"{currency}{value:,.2f}"


def format_number(value: float, decimals: int = 0) -> str:
    """
    Format large numbers with thousands separators

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        str: Formatted number string
    """
    if decimals == 0:
        return f"{value:,.0f}"
    else:
        return f"{value:,.{decimals}f}"


def generate_pdf_report(
    df_filtered: pd.DataFrame,
    summary_stats: dict,
    prediction_date: str,
    selected_products: list = None
) -> bytes:
    """
    Generate PDF report with filtered predictions and summary statistics

    Args:
        df_filtered: Filtered predictions dataframe (already formatted)
        summary_stats: Dictionary with summary metrics from results['summary']
        prediction_date: Date of the forecast
        selected_products: List of selected product types (for filter info)

    Returns:
        bytes: PDF file content
    """
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )
    from reportlab.lib.enums import TA_CENTER

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.5*inch, leftMargin=0.5*inch,
        topMargin=0.5*inch, bottomMargin=0.5*inch
    )

    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=18, textColor=colors.HexColor('#1f77b4'),
        alignment=TA_CENTER, spaceAfter=20
    )
    elements.append(Paragraph("Inventory Demand Forecast Report", title_style))

    # Metadata
    meta_text = f"<b>Forecast Date:</b> {prediction_date}<br/>"
    if selected_products:
        meta_text += f"<b>Filtered Products:</b> {', '.join(selected_products)}<br/>"
    meta_text += f"<b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elements.append(Paragraph(meta_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    # Summary Statistics Section
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor('#1f77b4'), spaceAfter=10
    )
    elements.append(Paragraph("Summary Statistics", heading_style))

    summary_data = [
        ['Metric', 'Value'],
        ['Total Predicted Demand (Q50)', format_number(summary_stats['total_predicted_demand_q50'])],
        ['Safety Stock Needed (Q90)', format_number(summary_stats['total_safety_stock_q90'])],
        ['SKUs with Expected Demand', f"{summary_stats['skus_with_demand']} / {summary_stats['total_skus']}"],
        ['Avg Manufacturing Lead Time', f"{summary_stats['avg_manufacturing_time']:.0f} days"]
    ]

    summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.4*inch))

    # Detailed Predictions Table
    elements.append(Paragraph("Detailed Predictions", heading_style))
    elements.append(Paragraph(f"Showing {len(df_filtered)} SKUs", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))

    # Limit to 100 rows
    max_rows = 100
    df_for_pdf = df_filtered.head(max_rows)

    if len(df_filtered) > max_rows:
        elements.append(Paragraph(
            f"<i>Note: Showing first {max_rows} of {len(df_filtered)} SKUs. "
            f"Download CSV for complete data.</i>",
            styles['Normal']
        ))
        elements.append(Spacer(1, 0.1*inch))

    # Build table data with formatted headers
    # Use multi-line headers for better readability
    formatted_headers = [
        'SKU',
        'Product\nType',
        'Manufacturing\nTime (days)',
        'Likelihood of\nDemand (%)',
        'Forecasted\nDemand (Median)',
        'Upper\nForecast (Q90)'
    ]
    table_data = [formatted_headers]
    for _, row in df_for_pdf.iterrows():
        table_data.append([str(val) for val in row.tolist()])

    # Create table
    col_widths = [1.0*inch, 1.2*inch, 1.3*inch, 1.3*inch, 1.5*inch, 1.2*inch]
    predictions_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    predictions_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    elements.append(predictions_table)

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
