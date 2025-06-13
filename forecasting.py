# 📦 Install required packages if you haven't already
# pip install streamlit plotly prophet sqlalchemy pymysql

import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from sqlalchemy import create_engine

# Setup DB connection
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT InvoiceDate, TotalPrice FROM cleaned_transactions", engine)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# Forecasting function
def build_forecast(df, periods=6):
    ts_df = df.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['TotalPrice'].sum().reset_index()
    ts_df.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y'

    model = Prophet()
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return ts_df, forecast, model

# Start of Streamlit app
st.title("📈 Sales Forecasting Dashboard")

# Load data
raw_df = load_data()

# Sidebar for forecast period
forecast_months = st.sidebar.slider("Months to Forecast", min_value=3, max_value=24, value=6)

# Build and show forecast
st.subheader("Historical Sales & Forecast")
ts_df, forecast, model = build_forecast(raw_df, periods=forecast_months)

# Merge for plotting
fig = px.line()
fig.add_scatter(x=ts_df['ds'], y=ts_df['y'], mode='lines', name='Actual Sales')
fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast')
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
fig.update_layout(title="Monthly Sales Forecast", xaxis_title="Date", yaxis_title="Sales Amount")

st.plotly_chart(fig)

# Optional: Show forecast table
if st.checkbox("Show Forecast Data"):
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_months))
