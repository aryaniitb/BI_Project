# 📦 Required packages:
# pip install streamlit plotly prophet pymysql sqlalchemy statsmodels

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
from statsmodels.nonparametric.smoothers_lowess import lowess
from sqlalchemy import create_engine

# 🔧 Streamlit config
st.set_page_config(page_title="Forecasting Dashboard", layout="wide")

# ✅ MySQL connection
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")

# ✅ Load from MySQL
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT InvoiceDate AS Date, Country, TotalPrice AS Revenue FROM cleaned_transactions", engine)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ---------------------------
# Sidebar controls
st.sidebar.header("Forecast Settings")
countries = sorted(df["Country"].dropna().unique())
country = st.sidebar.selectbox("Select Country", countries)
periods = st.sidebar.slider("Forecast horizon (months)", min_value=1, max_value=24, value=6)

# ---------------------------
# Filter by country
df_country = df[df["Country"] == country].copy()
monthly = df_country.groupby(pd.Grouper(key="Date", freq="MS")).sum(numeric_only=True).reset_index()
monthly = monthly.rename(columns={"Date": "ds", "Revenue": "y"})
monthly = monthly.dropna()

st.subheader(f"{country}: Forecasting Revenue")
st.write(f"📅 Available months: {len(monthly)}")

# ---------------------------
# Minimum check
if len(monthly) < 6:
    st.warning("❗ At least 6 months of data required.")
    st.stop()

# ---------------------------
# Smoothing
monthly["y_smoothed"] = lowess(monthly["y"], monthly["ds"], frac=0.25, return_sorted=False)

# Plot smoothed vs original
fig_smooth = go.Figure()
fig_smooth.add_trace(go.Scatter(x=monthly["ds"], y=monthly["y"], mode="lines+markers", name="Original"))
fig_smooth.add_trace(go.Scatter(x=monthly["ds"], y=monthly["y_smoothed"], mode="lines+markers", name="Smoothed"))
fig_smooth.update_layout(title="Smoothed Revenue Data", xaxis_title="Date", yaxis_title="Revenue")
st.plotly_chart(fig_smooth, use_container_width=True)

# ---------------------------
# Forecast button
if st.button("Run Forecast ▶️"):
    cap = monthly["y_smoothed"].max() * 1.10
    monthly["cap"] = cap

    history = monthly[["ds", "y_smoothed", "cap"]].rename(columns={"y_smoothed": "y"})

    # Prophet with logistic growth
    m = Prophet(
        growth="logistic",
        changepoint_prior_scale=0.1,
        seasonality_mode="multiplicative",
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(history)

    future = m.make_future_dataframe(periods=periods, freq="MS")
    future["cap"] = cap
    forecast = m.predict(future)

    # Plot forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["ds"], y=monthly["y"], mode="lines+markers", name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(color="orange")))

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=forecast["ds"].tolist() + forecast["ds"][::-1].tolist(),
        y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Uncertainty",
        showlegend=True
    ))

    fig.update_layout(
        title=f"{country}: Revenue Forecast",
        xaxis_title="Date",
        yaxis_title="Revenue",
        legend=dict(x=0, y=1.1, orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)
