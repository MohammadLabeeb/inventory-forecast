"""
Inventory Demand Forecasting Dashboard
Interactive Streamlit application for generating and visualizing inventory forecasts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import backend modules
from backend.forecaster import ForecastingPipeline
from backend.utils import (
    format_prediction_table,
    create_visualization_data,
    get_alert_skus,
    calculate_order_recommendations,
    format_number,
    generate_pdf_report
)

# Page configuration
st.set_page_config(
    page_title="Inventory Demand Forecasting",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


@st.cache_resource
def load_pipeline():
    """Load and cache the forecasting pipeline"""
    with st.spinner("Loading forecasting pipeline... This may take a minute."):
        pipeline = ForecastingPipeline()
        pipeline.prepare_data()
        pipeline.load_forecaster()
    return pipeline


def main():
    """Main application function"""

    # Header
    st.markdown('<h1 class="main-header">🏭 Inventory Demand Forecasting Dashboard</h1>',
                unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("⚙️ Forecast Configuration")

    # Fixed date for forecasting (simulating 2025-10-30 as "today")
    prediction_date = datetime(2025, 10, 30)

    # Display as static text in sidebar
    st.sidebar.markdown("### 📅 Current Date")
    st.sidebar.info("2025/10/30")
    st.sidebar.caption("Forecasts are generated for the current date: 2025-10-30")


    # Information about what the model predicts
    st.sidebar.info("""
    **What does the forecast predict?**

    The model predicts the **total cumulative demand** during each SKU's manufacturing lead time.

    For example, if a SKU has a 30-day manufacturing time, the forecast shows total units needed over those 30 days.
    """)

    # Generate forecast button
    if st.sidebar.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        generate_forecast(prediction_date)

    st.sidebar.markdown("---")

    # Information section
    with st.sidebar.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        This dashboard provides AI-powered inventory demand forecasts using a trained LightGBM model.

        **Features:**
        - Tier-specific predictions (5 tiers based on sales volume)
        - Lead-time aware forecasts (different for each SKU)
        - Safety stock recommendations (Q90 predictions)
        - Historical performance metrics (WAPE: 27.3%, Calibration: 86%)

        **What is predicted:**
        - **Q50 (Median)**: Expected cumulative demand over the manufacturing lead time
        - **Q90 (Upper Bound)**: Conservative safety stock estimate (90% confidence)

        **How it works:**
        Each SKU has a different manufacturing lead time (25-75 days). The model predicts total demand during that period.
        """)

    # Main content area
    if st.session_state.forecast_results is not None:
        display_results()
    else:
        display_welcome()


def get_sku_list():
    """Get list of available SKUs"""
    if st.session_state.pipeline is None or st.session_state.pipeline.df_features is None:
        return []

    skus = sorted(st.session_state.pipeline.df_features['SKU'].unique().tolist())
    return skus


def generate_forecast(prediction_date):
    """Generate forecast for all SKUs"""

    # Load pipeline if not already loaded
    if st.session_state.pipeline is None:
        st.session_state.pipeline = load_pipeline()

    # Generate forecast for all SKUs
    with st.spinner(f"Generating forecast for {prediction_date}..."):
        try:
            sku_filter = None  # Always forecast all SKUs

            results = st.session_state.pipeline.generate_forecast(
                prediction_date=pd.Timestamp(prediction_date),
                sku=sku_filter,
                horizon_days=1  # Single-day forecast only
            )

            st.session_state.forecast_results = results
            st.success(f"✅ Forecast generated successfully for {len(results['predictions'])} SKU(s)!")

        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")


def display_welcome():
    """Display welcome screen when no forecast has been generated"""
    st.info("👈 Click 'Generate Forecast' in the sidebar to begin.")

    st.markdown("""
    ### How It Works

    This dashboard predicts **cumulative demand during manufacturing lead time** for each SKU (A SKU, or stock keeping unit, is a unique, internal code assigned to a product for inventory management purposes).

    **Example:**
    - Current date: **2025-10-30**
    - For a SKU with **30-day manufacturing time**:
      - The model predicts total demand from **2025-10-31 to 2025-11-29**
    - For a SKU with **75-day manufacturing time**:
      - The model predicts total demand from **2025-10-31 to 2026-01-13**

    Each SKU gets a different forecast period based on its manufacturing lead time!
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Forecast Results")
        st.markdown("""
        - Summary metrics (Q50 and Q90)
        - Detailed prediction table
        - Export to CSV
        """)

    with col2:
        st.markdown("### 📈 Visualizations")
        st.markdown("""
        - Top SKUs by predicted demand
        - Product type breakdown
        - Tier distribution
        """)

    with col3:
        st.markdown("### 🎯 Model Performance")
        st.markdown("""
        - Accuracy metrics
        - Historical validation
        - Calibration analysis
        """)


def display_results():
    """Display forecast results in tabs"""

    results = st.session_state.forecast_results
    df_predictions = results['predictions']

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Forecast Results",
        "📈 Visualizations",
        "🎯 Model Performance",
        "🔍 SKU Deep Dive"
    ])

    with tab1:
        display_forecast_results(results, df_predictions)

    with tab2:
        display_visualizations(df_predictions)

    with tab3:
        display_model_performance()

    with tab4:
        display_sku_deep_dive(df_predictions)


def display_forecast_results(results, df_predictions):
    """Display forecast results and summary metrics"""

    st.subheader("📋 Forecast Summary")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Predicted Demand (Q50)",
            format_number(results['summary']['total_predicted_demand_q50']),
            help="Expected demand across all SKUs"
        )

    with col2:
        st.metric(
            "Safety Stock Needed (Q90)",
            format_number(results['summary']['total_safety_stock_q90']),
            help="Conservative safety stock estimate"
        )

    with col3:
        st.metric(
            "SKUs with Expected Demand",
            f"{results['summary']['skus_with_demand']} / {results['summary']['total_skus']}",
            help="Number of SKUs with non-zero predicted demand"
        )

    with col4:
        st.metric(
            "Avg Manufacturing Lead Time",
            f"{results['summary']['avg_manufacturing_time']:.0f} days",
            help="Average lead time across all SKUs"
        )

    st.markdown("---")

    # Prepare display-ready dataframe with renamed columns
    df_display = df_predictions.copy()

    # Round demand values to integers
    # Median: round to nearest integer (best estimate)
    # Q90: round up (safety stock buffer)
    df_display['pred_median'] = np.round(df_display['pred_median'])
    df_display['pred_upper'] = np.ceil(df_display['pred_upper'])

    # Convert prob_nonzero from decimal to percentage
    df_display['Likelihood of Demand (%)'] = (df_display['prob_nonzero'] * 100).round(2)

    # Rename columns
    df_display = df_display.rename(columns={
        'pred_median': 'Forecasted Demand (Median)',
        'pred_upper': 'Upper Forecast (Q90)',
        'manufacturing_time': 'Manufacturing Time (days)',
        'product_type': 'Product Type'
    })

    # Select columns to display (excluding tier and original prob_nonzero)
    display_columns = [
        'SKU',
        'Product Type',
        'Manufacturing Time (days)',
        'Likelihood of Demand (%)',
        'Forecasted Demand (Median)',
        'Upper Forecast (Q90)'
    ]
    df_display = df_display[display_columns]

    # Detailed prediction table in collapsible expander (minimized by default)
    with st.expander("📊 Detailed Predictions", expanded=False):
        # Product type filter
        available_products = sorted(df_predictions['product_type'].unique().tolist())

        selected_products = st.multiselect(
            "Filter by Product Type",
            options=available_products,
            default=available_products,
            help="Select one or more product types to filter the table"
        )

        # Apply filter
        if selected_products:
            df_filtered = df_display[df_display['Product Type'].isin(selected_products)]
        else:
            df_filtered = df_display

        # Show filtered count
        st.caption(f"Showing {len(df_filtered)} of {len(df_display)} SKUs")

        # Display table
        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=400
        )

        # Download buttons (side by side)
        col1, col2 = st.columns(2)

        with col1:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"forecast_{results['prediction_date']}.csv",
                mime="text/csv"
            )

        with col2:
            pdf_bytes = generate_pdf_report(
                df_filtered=df_filtered,
                summary_stats=results['summary'],
                prediction_date=str(results['prediction_date']),
                selected_products=selected_products if len(selected_products) < len(available_products) else None
            )

            st.download_button(
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name=f"forecast_{results['prediction_date']}.pdf",
                mime="application/pdf"
            )


def display_visualizations(df_predictions):
    """Display various visualizations"""

    st.subheader("📊 Forecast Visualizations")

    # Top 10 SKUs by predicted demand
    st.markdown("#### Top 10 SKUs by Predicted Demand")

    top_10 = df_predictions.nlargest(10, 'pred_median')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_10['SKU'],
        y=top_10['pred_median'],
        name='Forecast (Q50)',
        marker_color='steelblue'
    ))
    fig.add_trace(go.Bar(
        x=top_10['SKU'],
        y=top_10['pred_upper'],
        name='Safety Stock (Q90)',
        marker_color='orange'
    ))

    fig.update_layout(
        xaxis_title="SKU",
        yaxis_title="Predicted Demand (units)",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Product type breakdown
    st.markdown("#### Demand by Product Type")

    by_product = df_predictions.groupby('product_type')['pred_median'].sum().reset_index()

    fig = px.pie(
        by_product,
        values='pred_median',
        names='product_type',
        title='Predicted Demand Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)


def display_model_performance():
    """Display model performance metrics"""

    st.subheader("🎯 Model Performance Metrics")

    st.info("""
    📊 **How Metrics Are Calculated**

    This dashboard uses a **Hurdle Model** with 6 ML models working together:

    **Stage 1: Classification (Will demand occur?)**
    - 2 Binary Classifiers (one for high-volume tiers 1-2, one for low-volume tiers 3-4)
    - Predicts probability of any demand occurring

    **Stage 2: Regression (How much demand?)**
    - 4 Regressors total:
      - High-volume (Tiers 1-2): Q50 (median) and Q90 (upper bound)
      - Low-volume (Tiers 3-4): Q50 (median) and Q90 (upper bound)
    - Predict demand quantity if classifier says demand will occur

    **Final Prediction:** If classifier probability < 30%, forecast = 0. Otherwise, use regressor output.

    **Overall Metrics:** Weighted by product volume/sales to prioritize accuracy on high-impact SKUs.

    *Test Period: June - November 2025*
    """)

    # Regression Model Performance
    st.markdown("#### Regression Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "WAPE",
            "27.3%",
            help="Weighted Absolute Percentage Error - Measures forecast accuracy with higher weight on high-volume products. Lower is better. 27.3% means forecasts are on average 27.3% off from actuals."
        )

    with col2:
        st.metric(
            "R²",
            "0.86",
            help="Coefficient of Determination - Measures how much variance in demand is explained by the model. Range 0-1, higher is better. 0.86 means the model explains 86% of demand variation."
        )

    with col3:
        st.metric(
            "Calibration (Q90)",
            "86.0%",
            help="Safety Stock Coverage - Percentage of actual demand that falls below the Q90 forecast. Target: 90%. 86% means the upper bound is slightly conservative, covering 86% of actual demand cases."
        )

    with col4:
        st.metric(
            "MAE",
            "1.26",
            help="Mean Absolute Error - Average absolute difference between forecast and actual in units. Lower is better. 1.26 means forecasts are off by ~1.3 units on average."
        )

    # Classification Model Performance
    st.markdown("#### Classification Model Performance")
    st.caption("Binary classifiers predict whether demand will occur for each tier group")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "AUC",
            "0.89",
            help="Area Under ROC Curve - Measures classifier's ability to distinguish between demand/no-demand. Range 0-1, higher is better. 0.89 is excellent performance."
        )

    with col6:
        st.metric(
            "Accuracy",
            "84%",
            help="Overall classification accuracy - Percentage of correct demand/no-demand predictions across all products."
        )

    with col7:
        st.metric(
            "Precision",
            "81%",
            help="Of predictions that said 'demand will occur', 81% were correct. Higher means fewer false alarms (predicting demand when there is none)."
        )

    with col8:
        st.metric(
            "Recall",
            "76%",
            help="Of actual demand occurrences, we correctly predicted 76%. Higher means fewer missed opportunities (failing to predict demand when it occurs)."
        )

    st.markdown("---")

    # Performance by tier
    st.markdown("#### Performance by Product Tier")

    tier_performance = pd.DataFrame({
        'Tier': [1, 2, 3, 4],
        'WAPE (%)': [23.6, 22.3, 33.1, 30.5],
        'MAE': [3.49, 1.41, 1.03, 0.66],
        'Calibration (%)': [81.5, 86.9, 87.6, 84.7]
    })

    st.dataframe(tier_performance, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Feature Importance
    st.markdown("#### Feature Importance")
    st.caption("Top 15 most important features for demand prediction (High-Volume Products)")

    # Try to extract feature importance from loaded models
    try:
        if st.session_state.pipeline is not None:
            models = st.session_state.pipeline.forecaster.models
            high_vol_model = models['high_volume']['regressor_q50']

            # Extract feature importance
            importance = high_vol_model.feature_importance(importance_type='gain')
            feature_names = high_vol_model.feature_name()

            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(15)

            # Plot with Plotly
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model not loaded. Feature importance unavailable.")
    except Exception as e:
        st.warning(f"Could not extract feature importance: {str(e)}")

    st.markdown("---")

    # Model Architecture Diagram
    st.markdown("### 🔄 Model Architecture")
    st.caption("Visual representation of how the 6 ML models work together")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### High-Volume Products (Tiers 1-2)")
        st.info("""
        **Step 1:** Binary Classifier
        - Predicts: Will demand occur?
        - Output: Probability (0-1)
        - Metric: AUC

        **Step 2:** Hurdle Logic
        - If prob < 30% → Forecast = 0
        - If prob ≥ 30% → Continue to regression

        **Step 3:** Regression Models
        - Q50 Regressor → Median forecast
        - Q90 Regressor → Upper bound (safety stock)
        - Metric: Quantile loss
        """)

    with col2:
        st.markdown("#### Low-Volume Products (Tiers 3-4)")
        st.info("""
        **Step 1:** Binary Classifier
        - Predicts: Will demand occur?
        - Output: Probability (0-1)
        - More regularization than high-volume

        **Step 2:** Hurdle Logic
        - If prob < 30% → Forecast = 0
        - If prob ≥ 30% → Continue to regression

        **Step 3:** Regression Models
        - Q50 Regressor → Median forecast (regularized)
        - Q90 Regressor → Upper bound (regularized)
        - Higher regularization for stability
        """)

    st.success("""
    **💡 Key Insight:** Both tier groups use the same 3-stage approach (classifier + hurdle + regressors).
    The main difference is in model complexity: high-volume models use more complex trees (64 leaves),
    while low-volume models use simpler trees (31 leaves) with more regularization to prevent overfitting.
    """)


def display_sku_deep_dive(df_predictions):
    """Display detailed analysis for a single SKU"""

    st.subheader("🔍 SKU Deep Dive")

    # SKU selector
    if len(df_predictions['SKU'].unique()) == 1:
        # Single SKU already selected
        selected_sku = df_predictions['SKU'].iloc[0]
        st.info(f"Showing details for SKU: **{selected_sku}**")
    else:
        # Allow user to select a SKU
        selected_sku = st.selectbox(
            "Select SKU for detailed analysis",
            options=sorted(df_predictions['SKU'].unique())
        )

    # Get data for selected SKU
    sku_data = df_predictions[df_predictions['SKU'] == selected_sku]

    if len(sku_data) == 0:
        st.warning("No data available for selected SKU")
        return

    # Display SKU details
    row = sku_data.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Tier", row['tier'])

    with col2:
        st.metric("Product Type", row['product_type'])

    with col3:
        st.metric("Manufacturing Time", f"{row['manufacturing_time']:.0f} days")

    with col4:
        st.metric("Demand Probability", f"{row['prob_nonzero']:.1%}")

    st.markdown("---")

    # Forecast details
    st.markdown("#### Forecast Details")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Demand (Q50)", f"{int(row['pred_median'])} units")

    with col2:
        st.metric("Safety Stock (Q90)", f"{int(row['pred_upper'])} units")

    # Historical context
    if st.session_state.pipeline is not None:
        st.markdown("#### Historical Sales Pattern (Last 90 Days)")

        hist_data = st.session_state.pipeline.get_historical_data(selected_sku, days=90)

        if len(hist_data) > 0:
            fig = px.line(
                hist_data,
                x='date',
                y='quantity',
                title=f'Historical Sales for {selected_sku}',
                labels={'quantity': 'Daily Sales', 'date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical data available for this SKU")


if __name__ == "__main__":
    main()
