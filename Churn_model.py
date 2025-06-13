import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load customer summary from MySQL
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")
df = pd.read_sql("SELECT * FROM customer_summary", engine)

# 2) Prepare feature matrix X and target y
features = ["NumOrders", "TotalQuantity", "TotalSpent"]  # Removed RecencyDays
X = df[features]
y = df["Churn"]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Train a RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5) Predict & evaluate
y_pred = model.predict(X_test)
print("▶️ Accuracy:", accuracy_score(y_test, y_pred))
print("\n▶️ Classification Report:\n", classification_report(y_test, y_pred))

# 6) Feature importance plot
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importances for Churn Model")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()
print("🔍 Churn Class Distribution:")
print(df['Churn'].value_counts())
print("Train set churn distribution:\n", y_train.value_counts())
print("Test set churn distribution:\n", y_test.value_counts())
print(df.head(10))
print(df[features + ['Churn']].corr())
