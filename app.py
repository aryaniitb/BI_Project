import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# MySQL connection
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM cleaned_transactions", engine)

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("🔍 Filters")
st.title("📊 Smart Business Intelligence Dashboard")

countries = df['Country'].unique()
years = sorted(df['InvoiceYear'].unique())
months = sorted(df['InvoiceMonth'].unique())

selected_country = st.sidebar.selectbox("🌍 Country", countries)
selected_year = st.sidebar.selectbox("📅 Year", years)
selected_month = st.sidebar.selectbox("🗓️ Month", months)

filtered_df = df[(df['Country'] == selected_country) &
                 (df['InvoiceYear'] == selected_year) &
                 (df['InvoiceMonth'] == selected_month)]

# --- KPIs ---
st.markdown("### 🔑 Key Performance Indicators")
total_revenue = filtered_df['TotalPrice'].sum()
total_orders = filtered_df['InvoiceNo'].nunique()
total_products = filtered_df['StockCode'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("📈 Total Revenue", f"£{total_revenue:,.2f}")
col2.metric("🧾 Total Orders", total_orders)
col3.metric("📦 Unique Products", total_products)

# --- Monthly Revenue Trend ---
st.markdown("### 📆 Monthly Revenue Trend")
monthly_revenue = (
    df[df['Country'] == selected_country]
    .groupby(['InvoiceYear', 'InvoiceMonth'])['TotalPrice']
    .sum()
    .reset_index()
)
fig1 = px.line(monthly_revenue, x='InvoiceMonth', y='TotalPrice', color='InvoiceYear',
               title=f"Monthly Revenue in {selected_country}")
st.plotly_chart(fig1, use_container_width=True)

# --- Top 10 Products ---
st.markdown("### 🏆 Top 10 Products by Revenue")
top_products = (
    filtered_df.groupby("Description")["TotalPrice"]
    .sum().sort_values(ascending=False).head(10).reset_index()
)
fig2 = px.bar(top_products, x="TotalPrice", y="Description", orientation='h',
              title="Top 10 Best-Selling Products")
st.plotly_chart(fig2, use_container_width=True)

# --- Revenue by Day of Week ---
st.markdown("### 📅 Revenue by Day of the Week")
dow = (
    filtered_df.groupby("DayOfWeek")["TotalPrice"]
    .sum().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    .reset_index()
)
fig3 = px.bar(dow, x="DayOfWeek", y="TotalPrice", color="DayOfWeek",
              title="Revenue by Day of the Week")
st.plotly_chart(fig3, use_container_width=True)

# --- Revenue by Hour ---
st.markdown("### 🕒 Revenue by Hour of the Day")
hourly = filtered_df.groupby("InvoiceHour")["TotalPrice"].sum().reset_index()
fig4 = px.line(hourly, x="InvoiceHour", y="TotalPrice",
               title="Hourly Revenue Trends")
st.plotly_chart(fig4, use_container_width=True)

# --- Country-wise Comparison ---
st.markdown("### 🌍 Revenue Comparison Across Countries")
country_compare = (
    df[(df['InvoiceYear'] == selected_year) & (df['InvoiceMonth'] == selected_month)]
    .groupby("Country")["TotalPrice"]
    .sum().sort_values(ascending=False).reset_index().head(10)
)
fig5 = px.bar(country_compare, x="Country", y="TotalPrice",
              title="Top Countries by Revenue in Selected Month")
st.plotly_chart(fig5, use_container_width=True)

# --- Order Volume by Country ---
st.markdown("### 🚚 Number of Orders by Country")
order_vol = (
    df[(df['InvoiceYear'] == selected_year) & (df['InvoiceMonth'] == selected_month)]
    .groupby("Country")["InvoiceNo"]
    .nunique().reset_index().sort_values(by="InvoiceNo", ascending=False).head(10)
)
fig6 = px.bar(order_vol, x="Country", y="InvoiceNo",
              title="Top 10 Countries by Order Volume")
st.plotly_chart(fig6, use_container_width=True)

# --- Raw Data Table ---
st.markdown("### 📄 Sample Transaction Data")
st.dataframe(filtered_df.head(50))

st.success("✅ Dashboard loaded successfully. More features like forecasting, recommendation and NLP-based queries can be added next!")
