# recommendation_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sqlalchemy import create_engine

# ------------------ 1. MySQL Connection ------------------
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")

# ------------------ 2. Load Product Descriptions ------------------
query = """
SELECT DISTINCT Description
FROM cleaned_transactions
WHERE Description IS NOT NULL AND LENGTH(TRIM(Description)) > 0;
"""
df = pd.read_sql(query, engine)

# ------------------ 3. Clean Text ------------------
df['Description'] = df['Description'].str.strip().str.lower()
df = df.drop_duplicates(subset=['Description'])
print("\n📝 Sample product descriptions:")
print(df['Description'].sample(10).tolist())

# ------------------ 4. TF-IDF + Similarity ------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
cos_sim = cosine_similarity(tfidf_matrix)

# Map index to product
indices = pd.Series(df.index, index=df['Description'])

# ------------------ 5. Recommendation Function ------------------
def get_recommendations(product_name, top_n=5):
    product_name = product_name.strip().lower()
    if product_name not in indices:
        return f"❌ '{product_name}' not found in product list."
    
    idx = indices[product_name]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return df['Description'].iloc[top_indices].tolist()

# ------------------ 6. Example & Save ------------------
if __name__ == "__main__":
    test_product = "ivory kitchen scales"  # Use real product from your DB
    print(f"🧥 Recommendations for: '{test_product}'")
    print(get_recommendations(test_product))

    # ✅ Ensure directory exists
    import os
    os.makedirs("models", exist_ok=True)

    # ✅ Save the model
    with open("models/product_recommender.pkl", "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix, df, indices), f)
