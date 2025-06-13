# nlp_assistant.py

import streamlit as st

st.set_page_config(page_title="Retail Intelligence Assistant", layout="wide")
st.title("💬 Ask Me Anything (Retail Assistant)")

query = st.text_input("🔎 Ask a question (e.g. 'Forecast revenue for France', 'Show churn risk')")

if query:
    query_lower = query.lower()

    if "forecast" in query_lower or "revenue" in query_lower:
        st.info("🔁 Forwarding to Forecasting Module... (You can embed a function here)")
        # Example: import forecasting_dashboard; forecasting_dashboard.run()

    elif "churn" in query_lower or "customer" in query_lower:
        st.info("⚠️ Forwarding to Churn Module...")
        # Example: import churn_dashboard; churn_dashboard.run()

    elif "recommend" in query_lower or "product" in query_lower:
        st.info("🎯 Forwarding to Recommendation Module...")
        # Example: import recommender; recommender.run()

    else:
        st.warning("❌ Sorry, I couldn't understand. Try asking about revenue, churn, or recommendations.")
