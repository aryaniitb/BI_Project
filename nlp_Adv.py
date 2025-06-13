import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine

from huggingface_hub import login
login(token="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")

# Set device safely
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- DATABASE ----------
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")

# ---------- TF-IDF RECOMMENDATION ----------
desc_df = pd.read_sql("""
    SELECT DISTINCT Description
    FROM cleaned_transactions
    WHERE Description IS NOT NULL AND LENGTH(TRIM(Description)) > 0;
""", engine)

desc_df['Description'] = desc_df['Description'].str.strip().str.lower()
desc_df = desc_df.drop_duplicates(subset=['Description'])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(desc_df['Description'])
cos_sim = cosine_similarity(tfidf_matrix)
indices = pd.Series(desc_df.index, index=desc_df['Description'])

def get_recommendations(product_name, top_n=5):
    product_name = product_name.strip().lower()
    if product_name not in indices:
        return f"❌ '{product_name}' not found in product list."
    idx = indices[product_name]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return desc_df['Description'].iloc[top_indices].tolist()

# ---------- FORECASTING ----------
def run_forecast_pipeline():
    from prophet import Prophet
    from statsmodels.nonparametric.smoothers_lowess import lowess

    df = pd.read_sql("SELECT InvoiceDate AS Date, Country, TotalPrice AS Revenue FROM cleaned_transactions", engine)
    df["Date"] = pd.to_datetime(df["Date"])
    df_country = df[df["Country"] == "United Kingdom"].copy()

    monthly = df_country.groupby(pd.Grouper(key="Date", freq="MS")).sum(numeric_only=True).reset_index()
    monthly = monthly.rename(columns={"Date": "ds", "Revenue": "y"}).dropna()
    if len(monthly) < 6:
        return "❗ Not enough data to forecast (need at least 6 months)."

    monthly["y_smoothed"] = lowess(monthly["y"], monthly["ds"], frac=0.25, return_sorted=False)
    cap = monthly["y_smoothed"].max() * 1.10
    monthly["cap"] = cap
    history = monthly[["ds", "y_smoothed", "cap"]].rename(columns={"y_smoothed": "y"})

    model = Prophet(growth="logistic", seasonality_mode="multiplicative")
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(history)

    future = model.make_future_dataframe(periods=6, freq="MS")
    future["cap"] = cap
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6)

# ---------- CHURN PREDICTION ----------
def run_churn_pipeline():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.utils import class_weight
    import numpy as np

    df = pd.read_sql("SELECT * FROM customer_summary", engine)
    features = ["NumOrders", "TotalQuantity", "TotalSpent"]
    X = df[features]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    class_weights = {0: weights[0], 1: weights[1]}

    model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "churn_precision": report["1"]["precision"],
        "churn_recall": report["1"]["recall"],
        "churn_f1": report["1"]["f1-score"]
    }

# ---------- LLM LOADING ----------
def load_model(model_name="distilgpt2"):
    try:
        print("⏬ Loading tokenizer and model (CPU only)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        return tokenizer, model
    except Exception as e:
        print(f"⚠️ Failed to load {model_name}. Error: {e}")
        return None, None

def ask_llm(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.95, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- INTENT DETECTION ----------
def identify_intent(user_input):
    user_input = user_input.lower()
    if "recommend" in user_input or "similar" in user_input:
        return "recommendation"
    elif "forecast" in user_input or "predict" in user_input:
        return "forecasting"
    elif "churn" in user_input or "loss" in user_input:
        return "churn"
    else:
        return "unknown"

# ---------- MAIN CHATBOT ----------
def main():
    print("🤖 Welcome to the BI Chatbot!")
    tokenizer, model = load_model()

    while True:
        user_input = input("\n🔎 Ask a question (or type 'exit' to quit):\n> ")
        if user_input.strip().lower() == "exit":
            print("👋 Bye!")
            break

        intent = identify_intent(user_input)

        if intent == "recommendation":
            product_name = input("Enter product description:\n> ")
            recs = get_recommendations(product_name)
            explanation = ask_llm(f"Explain why these products are similar to '{product_name}': {recs}", tokenizer, model)
            print("\n📦 Recommendations:", recs)
            print("\n🧠 LLM Says:", explanation)

        elif intent == "forecasting":
            forecast_df = run_forecast_pipeline()
            explanation = ask_llm(f"Summarize the forecasted revenue data:\n{forecast_df}", tokenizer, model)
            print("\n📈 Forecast Data:\n", forecast_df)
            print("\n🧠 LLM Summary:", explanation)

        elif intent == "churn":
            churn_results = run_churn_pipeline()
            explanation = ask_llm(f"Interpret these churn prediction metrics:\n{churn_results}", tokenizer, model)
            print("\n🚪 Churn Prediction:\n", churn_results)
            print("\n🧠 LLM Explanation:", explanation)

        else:
            response = ask_llm(f"User asked: {user_input}. Try to provide a helpful BI response.", tokenizer, model)
            print("\n🧠 LLM Response:", response)

if __name__ == "__main__":
    main()
