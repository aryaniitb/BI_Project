# churn_dashboard.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ Config ------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="centered")
st.title("⚠️ Customer Churn Prediction")

# ------------------ Load Data ------------------
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")
df = pd.read_sql("SELECT * FROM customer_summary", engine)

features = ["NumOrders", "TotalQuantity", "TotalSpent"]
X = df[features]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# ------------------ Train Model ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------ Feature Input ------------------
st.subheader("🔍 Predict for Custom Customer")
input_data = {}
for col in features:
    val = st.number_input(f"{col}", min_value=0.0, value=float(X[col].mean()))
    input_data[col] = val

if st.button("Predict Churn"):
    df_input = pd.DataFrame([input_data])
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    st.success(f"Prediction: {'Churn' if pred else 'Not Churn'} (Churn Risk = {prob:.2%})")

# ------------------ Feature Importance ------------------
st.subheader("📊 Feature Importance")
importances = pd.Series(model.feature_importances_, index=features).sort_values()
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importances")
st.pyplot(plt.gcf())
