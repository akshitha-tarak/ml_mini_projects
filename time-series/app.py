import os
import sys

# Ensure local virtual environment packages can be loaded if run in various setups
_venv_site = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages"))
if os.path.exists(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

# Page configuration
st.set_page_config(
    page_title="Airline Passenger Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.5);
    }
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-right: 6px;
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #a855f7;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .header-box {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(147, 51, 234, 0.08) 100%);
        border: 1px solid rgba(147, 51, 234, 0.2);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "airline-passengers.csv")
    df = pd.read_csv(csv_path)
    df["Month"] = pd.to_datetime(df["Month"])
    return df

@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model.joblib")
    full_path = os.path.join(base, "model_full.joblib")
    
    test_model = joblib.load(model_path) if os.path.exists(model_path) else None
    full_model = joblib.load(full_path) if os.path.exists(full_path) else None
    return test_model, full_model

df = load_data()
test_model, full_model = load_models()

# Fallback in case artifacts haven't been exported yet
if test_model is None or full_model is None:
    st.warning("⚠️ Serialized models not found. Training models dynamically...")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train_log = np.log(df.iloc[:-12]["Passengers"])
    test_model = SARIMAX(train_log, order=[1, 1, 1], seasonal_order=(1, 1, 1, 12)).fit(disp=False, maxiter=200)
    full_log = np.log(df["Passengers"])
    full_model = SARIMAX(full_log, order=[1, 1, 1], seasonal_order=(1, 1, 1, 12)).fit(disp=False, maxiter=200)

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/airplane-take-off.png", width=70)
    st.title("Forecast Controls")
    
    mode = st.radio(
        "Forecast Operation Mode",
        ["🚀 Future Projections (1961+)", "🎯 Backtest Validation (1960 Test Set)"],
        index=0
    )
    
    st.markdown("---")
    
    if mode.startswith("🚀 Future"):
        horizon = st.slider("Forecasting Horizon (Months Ahead)", min_value=1, max_value=36, value=12, step=1)
    else:
        horizon = 12
        st.info("Validation mode evaluates the last 12 months (1960) against recorded actuals.")
    
    confidence_level = st.select_slider(
        "Prediction Confidence Interval",
        options=[80, 90, 95],
        value=95,
        format_func=lambda x: f"{x}% Confidence Interval"
    )
    alpha = 1.0 - (confidence_level / 100.0)

    st.markdown("---")
    st.subheader("Filter & Scenario")
    history_view = st.selectbox(
        "Historical Window Display",
        ["All History (1949–1960)", "Last 5 Years (1956–1960)", "Last 3 Years (1958–1960)"]
    )
    
    shock_pct = st.slider(
        "Simulate Demand Shock / Growth",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        help="Simulate macroeconomic or seasonal shock impact on future forecasted demand"
    )
    
    st.markdown("---")
    st.caption("Machine Learning Mini-Projects Monorepo")
    st.caption("Airline Passenger Time Series Pipeline")

# Header Section
st.markdown("""
<div class="header-box">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">✈️ Airline Passenger Demand Forecasting</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.85; font-size: 1rem;">
                Production-grade time series forecasting powered by SARIMAX with multiplicative seasonal dynamics.
            </p>
        </div>
        <div>
            <span class="badge">SARIMAX (1,1,1)x(1,1,1,12)</span>
            <span class="badge badge-purple">Log-Transformed</span>
            <span class="badge badge-green">RMSE: 16.66</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Prepare Time Series Forecast
if mode.startswith("🚀 Future"):
    # Forward forecasting using model trained on full dataset
    active_model = full_model
    forecast_res = active_model.get_forecast(steps=horizon)
    fc_log = forecast_res.predicted_mean
    conf_log = forecast_res.conf_int(alpha=alpha)
    
    # Exponentiate back to passenger counts
    multiplier = 1.0 + (shock_pct / 100.0)
    fc_values = np.exp(fc_log) * multiplier
    lower_bound = np.exp(conf_log.iloc[:, 0]) * multiplier
    upper_bound = np.exp(conf_log.iloc[:, 1]) * multiplier
    
    # Build future month timestamps
    last_date = df["Month"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
    
    df_forecast = pd.DataFrame({
        "Month": future_dates,
        "Passengers": fc_values.values,
        "Lower_CI": lower_bound.values,
        "Upper_CI": upper_bound.values,
        "Type": "Forecast"
    })
    
    last_actual = df["Passengers"].iloc[-1]
    first_pred = df_forecast["Passengers"].iloc[0]
    next_mom_growth = ((first_pred - last_actual) / last_actual) * 100.0
    peak_pred_row = df_forecast.loc[df_forecast["Passengers"].idxmax()]
    mean_forecast = df_forecast["Passengers"].mean()
    
else:
    # Validation mode (test set 1960)
    active_model = test_model
    forecast_res = active_model.get_forecast(steps=12)
    fc_log = forecast_res.predicted_mean
    conf_log = forecast_res.conf_int(alpha=alpha)
    
    fc_values = np.exp(fc_log)
    lower_bound = np.exp(conf_log.iloc[:, 0])
    upper_bound = np.exp(conf_log.iloc[:, 1])
    
    test_actuals = df.iloc[-12:].copy()
    future_dates = test_actuals["Month"].values
    
    df_forecast = pd.DataFrame({
        "Month": future_dates,
        "Passengers": fc_values.values,
        "Lower_CI": lower_bound.values,
        "Upper_CI": upper_bound.values,
        "Type": "Forecast (Validation)"
    })
    
    val_mse = mean_squared_error(test_actuals["Passengers"], fc_values)
    val_mae = mean_absolute_error(test_actuals["Passengers"], fc_values)
    val_rmse = root_mean_squared_error(test_actuals["Passengers"], fc_values)
    
    last_actual = df.iloc[-13]["Passengers"]
    first_pred = df_forecast["Passengers"].iloc[0]
    next_mom_growth = ((first_pred - last_actual) / last_actual) * 100.0
    peak_pred_row = df_forecast.loc[df_forecast["Passengers"].idxmax()]
    mean_forecast = df_forecast["Passengers"].mean()

# Key Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_str = f"{next_mom_growth:+.1f}% vs last actual"
    st.metric(
        label="Next Month Projected",
        value=f"{int(round(first_pred)):,} pax",
        delta=delta_str
    )

with col2:
    st.metric(
        label="Peak Forecast Month",
        value=f"{int(round(peak_pred_row['Passengers'])):,} pax",
        delta=f"{peak_pred_row['Month'].strftime('%b %Y')}"
    )

with col3:
    st.metric(
        label="Horizon Average Demand",
        value=f"{int(round(mean_forecast)):,} pax/mo",
        delta=f"{horizon} months"
    )

with col4:
    if mode.startswith("🚀 Future"):
        st.metric(
            label="Validated Model RMSE",
            value="16.66 pax",
            delta="MAE: 11.99 | MSE: 277.41",
            delta_color="normal"
        )
    else:
        st.metric(
            label="Test Set RMSE (1960)",
            value=f"{val_rmse:.2f} pax",
            delta=f"MAE: {val_mae:.2f} | MSE: {val_mse:.2f}",
            delta_color="normal"
        )

st.write("")

# Prepare Filtered Historical Data for Chart
if history_view == "Last 3 Years (1958–1960)":
    plot_df = df[df["Month"] >= "1958-01-01"].copy()
elif history_view == "Last 5 Years (1956–1960)":
    plot_df = df[df["Month"] >= "1956-01-01"].copy()
else:
    plot_df = df.copy()

plot_df["Type"] = "Actual (Historical)"

# Main Interactive Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Interactive Forecast",
    "🔬 Seasonal Decomposition",
    "📋 Forecast Table & Export",
    "⚙️ Model Architecture & Specs"
])

with tab1:
    st.subheader("Historical Trajectory vs Future Projection")
    
    # Altair Interactive Plot
    # 1. Historical Actual Line
    hist_chart = alt.Chart(plot_df).mark_line(
        color="#3b82f6",
        strokeWidth=2.5
    ).encode(
        x=alt.X("Month:T", title="Date", axis=alt.Axis(format="%b %Y", labelAngle=-30)),
        y=alt.Y("Passengers:Q", title="Monthly Passengers (in thousands)"),
        tooltip=[
            alt.Tooltip("Month:T", format="%B %Y", title="Month"),
            alt.Tooltip("Passengers:Q", format=",.1f", title="Actual Passengers"),
            alt.Tooltip("Type:N", title="Category")
        ]
    )
    
    # 2. Historical Points
    hist_points = alt.Chart(plot_df).mark_circle(size=30, color="#3b82f6").encode(
        x="Month:T",
        y="Passengers:Q",
        tooltip=[
            alt.Tooltip("Month:T", format="%B %Y", title="Month"),
            alt.Tooltip("Passengers:Q", format=",.1f", title="Actual Passengers")
        ]
    )
    
    # 3. Forecast Line
    fc_chart = alt.Chart(df_forecast).mark_line(
        color="#f59e0b",
        strokeWidth=3,
        strokeDash=[5, 5] if mode.startswith("🎯") else [0]
    ).encode(
        x="Month:T",
        y="Passengers:Q",
        tooltip=[
            alt.Tooltip("Month:T", format="%B %Y", title="Forecast Month"),
            alt.Tooltip("Passengers:Q", format=",.1f", title="Predicted Passengers"),
            alt.Tooltip("Lower_CI:Q", format=",.1f", title=f"Lower {confidence_level}%"),
            alt.Tooltip("Upper_CI:Q", format=",.1f", title=f"Upper {confidence_level}%")
        ]
    )
    
    fc_points = alt.Chart(df_forecast).mark_point(
        size=40,
        color="#f59e0b",
        filled=True
    ).encode(
        x="Month:T",
        y="Passengers:Q",
        tooltip=[
            alt.Tooltip("Month:T", format="%B %Y", title="Forecast Month"),
            alt.Tooltip("Passengers:Q", format=",.1f", title="Predicted Passengers")
        ]
    )
    
    # 4. Confidence Interval Band
    band_chart = alt.Chart(df_forecast).mark_area(
        opacity=0.22,
        color="#f59e0b"
    ).encode(
        x="Month:T",
        y="Lower_CI:Q",
        y2="Upper_CI:Q"
    )
    
    # If in validation mode, also display actual test points for direct visual comparison
    if mode.startswith("🎯 Validation"):
        test_points = alt.Chart(test_actuals).mark_point(
            color="#ef4444",
            size=55,
            shape="diamond"
        ).encode(
            x="Month:T",
            y="Passengers:Q",
            tooltip=[
                alt.Tooltip("Month:T", format="%B %Y", title="Test Month"),
                alt.Tooltip("Passengers:Q", format=",.1f", title="Ground Truth Actual")
            ]
        )
        combined_chart = (hist_chart + hist_points + band_chart + fc_chart + fc_points + test_points)
    else:
        combined_chart = (hist_chart + hist_points + band_chart + fc_chart + fc_points)
    
    combined_chart = combined_chart.properties(
        width="container",
        height=430
    ).interactive()
    
    st.altair_chart(combined_chart, use_container_width=True)
    
    # Legend guide & Insights card
    c_leg1, c_leg2 = st.columns([1.5, 2.5])
    with c_leg1:
        st.markdown("""
        **Chart Legend:**
        - <span style="color: #3b82f6; font-weight: bold;">● Solid Blue</span>: Historical Actual Passenger Volume
        - <span style="color: #f59e0b; font-weight: bold;">● Orange</span>: SARIMAX Predicted Future Volume
        - <span style="background: rgba(245, 158, 11, 0.25); padding: 2px 6px; border-radius: 4px; font-weight: bold; color: #d97706;">Shaded Area</span>: {conf}% Confidence Interval
        """ + ("""
        - <span style="color: #ef4444; font-weight: bold;">◆ Red Diamonds</span>: Ground Truth Test Actuals (1960)
        """ if mode.startswith("🎯") else ""), unsafe_allow_html=True)
        
    with c_leg2:
        if shock_pct != 0:
            st.info(f"⚡ Scenario active: Applied a **{shock_pct:+d}%** simulated demand adjustment to all forecast steps.")
        else:
            st.success("✅ Baseline projection generated under standard historical seasonality and post-war aviation growth trend.")

with tab2:
    st.subheader("Seasonal Time Series Decomposition (Multiplicative)")
    st.write("Decomposing the raw airline passenger time series into its underlying structural drivers:")
    
    # Calculate seasonal decomposition
    ts_series = df.set_index("Month")["Passengers"]
    decomp = seasonal_decompose(ts_series, model="multiplicative", period=12)
    
    decomp_df = pd.DataFrame({
        "Month": df["Month"],
        "Observed": decomp.observed.values,
        "Trend": decomp.trend.values,
        "Seasonal": decomp.seasonal.values,
        "Residual": decomp.resid.values
    })
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**1. Underlying Long-Term Trend**")
        trend_chart = alt.Chart(decomp_df.dropna(subset=["Trend"])).mark_line(color="#2563eb", strokeWidth=2.5).encode(
            x=alt.X("Month:T", title="Year"),
            y=alt.Y("Trend:Q", title="Trend"),
            tooltip=["Month:T", "Trend:Q"]
        ).properties(height=200)
        st.altair_chart(trend_chart, use_container_width=True)
        st.caption("Steady upward expansion driven by global commercial airline adoption between 1949 and 1960.")
        
    with col_t2:
        st.markdown("**2. Seasonal Multiplier Cycle (12-Month Rhythm)**")
        season_chart = alt.Chart(decomp_df).mark_line(color="#10b981", strokeWidth=2.5).encode(
            x=alt.X("Month:T", title="Year"),
            y=alt.Y("Seasonal:Q", title="Seasonal Index"),
            tooltip=["Month:T", "Seasonal:Q"]
        ).properties(height=200)
        st.altair_chart(season_chart, use_container_width=True)
        st.caption("Predictable summer peaks in July–August (+30% above trend) and trough in November (-20%).")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**3. Irregular / Residual Variations**")
        resid_chart = alt.Chart(decomp_df.dropna(subset=["Residual"])).mark_circle(color="#f97316", size=25).encode(
            x=alt.X("Month:T", title="Year"),
            y=alt.Y("Residual:Q", title="Residual Multiplier"),
            tooltip=["Month:T", "Residual:Q"]
        ).properties(height=200)
        st.altair_chart(resid_chart, use_container_width=True)
        st.caption("Residual variance remains centered near 1.0, confirming clean model separability.")
        
    with col_r2:
        st.markdown("**4. Augmented Dickey-Fuller (ADF) Stationarity Analysis**")
        st.markdown("""
        - **ADF Statistic**: `0.815`
        - **P-Value**: `0.992` (Non-Stationary in raw levels)
        - **Remedy**: Natural log transformation + First differencing ($d=1, D=1, s=12$)
        - **Result**: Variance stabilized across the decade, fulfilling SARIMAX stationarity assumptions.
        """)

with tab3:
    st.subheader(f"Detailed Forecast Summary ({horizon} Months)")
    
    export_df = df_forecast.copy()
    export_df["Month_Str"] = export_df["Month"].dt.strftime("%Y-%m")
    export_df["Forecast_Pax"] = export_df["Passengers"].round(1)
    export_df["Lower_Pax"] = export_df["Lower_CI"].round(1)
    export_df["Upper_Pax"] = export_df["Upper_CI"].round(1)
    
    # Calculate MoM growth %
    export_df["MoM_Growth_%"] = export_df["Forecast_Pax"].pct_change().fillna((first_pred - last_actual)/last_actual) * 100.0
    export_df["MoM_Growth_%"] = export_df["MoM_Growth_%"].round(2)
    
    display_df = export_df[["Month_Str", "Forecast_Pax", "Lower_Pax", "Upper_Pax", "MoM_Growth_%"]].rename(
        columns={
            "Month_Str": "Month",
            "Forecast_Pax": "Projected Passengers (k)",
            "Lower_Pax": f"Lower Bound ({confidence_level}%)",
            "Upper_Pax": f"Upper Bound ({confidence_level}%)",
            "MoM_Growth_%": "MoM Growth (%)"
        }
    )
    
    st.dataframe(
        display_df.style.format({
            "Projected Passengers (k)": "{:,.1f}",
            "Lower Bound ({}%)".format(confidence_level): "{:,.1f}",
            "Upper Bound ({}%)".format(confidence_level): "{:,.1f}",
            "MoM Growth (%)": "{:+.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast Schedule (CSV)",
        data=csv_bytes,
        file_name=f"airline_passenger_forecast_{horizon}mo.csv",
        mime="text/csv",
        type="primary"
    )

with tab4:
    st.subheader("Model Specifications & Architecture")
    
    st.markdown("""
    ### 📐 Mathematical Formulation:
    The pipeline models the logarithm of monthly passenger volume using a Seasonal Autoregressive Integrated Moving Average (**SARIMAX**) architecture:
    
    $$\\text{SARIMAX}(p=1, d=1, q=1) \\times (P=1, D=1, Q=1)_{s=12}$$
    
    Where:
    - **$y_t = \\ln(\\text{Passengers}_t)$**: Logarithmic transform ensures constant residual variance across heteroscedastic growth.
    - **Non-Seasonal Orders $(1, 1, 1)$**:
      - $p=1$: Autoregressive lag term captures previous month dynamics.
      - $d=1$: First-order differencing removes secular trend.
      - $q=1$: Moving average captures short-term shocks.
    - **Seasonal Orders $(1, 1, 1)_{12}$**:
      - $P=1$: Seasonal AR accounts for annual correlation with same month in previous year.
      - $D=1$: Seasonal differencing eliminates repeating 12-month cyclical pattern.
      - $Q=1$: Seasonal MA smooths annual residual variations.
      - $s=12$: Monthly annual seasonality cycle.
    
    ---
    
    ### 🏆 Benchmark Evaluation on 1960 Test Split:
    """)
    
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.info("**Mean Squared Error (MSE)**\n# 277.41")
    with mcol2:
        st.success("**Mean Absolute Error (MAE)**\n# 11.99")
    with mcol3:
        st.info("**Root Mean Squared Error (RMSE)**\n# 16.66")
        
    st.markdown("""
    > **Reproducibility Note:** All parameters and model artifacts strictly correspond to the training pipeline established in `time-series/timefore.ipynb` and serialized via `time-series/export_artifacts.py`.
    """)
