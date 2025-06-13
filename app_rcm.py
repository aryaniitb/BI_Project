# app.py

import streamlit as st
import pickle

# ---------------- Load Model ----------------
with open("models/product_recommender.pkl", "rb") as f:
    vectorizer, tfidf_matrix, df, indices = pickle.load(f)

# ---------------- Recommendation Function ----------------
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(product_name, top_n=5):
    product_name = product_name.strip().lower()
    if product_name not in indices:
        return None
    idx = indices[product_name]
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return df['Description'].iloc[top_indices].tolist()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🧠 Product Recommender", layout="wide")
st.title("🛍️ Smart Product Recommendation")

# Dropdown for selecting a product
selected_product = st.selectbox("Select a Product", sorted(df['Description'].unique()))

if st.button("🔍 Recommend Similar Products"):
    recommendations = get_recommendations(selected_product)
    if recommendations:
        st.success(f"Recommendations for: **{selected_product}**")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.error("❌ Product not found. Try another.")
