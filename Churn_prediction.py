import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load Data from MySQL
# ---------------------------
engine = create_engine("mysql+pymysql://root:root@localhost/retail_data")
df = pd.read_sql("SELECT * FROM customer_summary", engine)

# ---------------------------
# 2. Prepare features and labels
# ---------------------------
features = ["NumOrders", "TotalQuantity", "TotalSpent"]
X = df[features]
y = df["Churn"]

# ---------------------------
# 3. Train-Test Split (Stratified)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🔍 Churn Class Distribution:")
print(df['Churn'].value_counts())
print("\nTrain set churn distribution:\n", y_train.value_counts())
print("Test set churn distribution:\n", y_test.value_counts())

# ---------------------------
# 4. Class Weight Calculation
# ---------------------------
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print("\n📊 Class Weights:", class_weights)

# ---------------------------
# 5. Model Training
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
model.fit(X_train, y_train)

# ---------------------------
# 6. Predictions and Evaluation
# ---------------------------
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

print("\n▶️ Accuracy:", accuracy_score(y_test, y_pred))
print("\n▶️ Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 7. Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active (0)", "Churned (1)"])
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------
# 8. Feature Importances
# ---------------------------
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Feature Importances")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# ---------------------------
# 9. ROC Curve and AUC
# ---------------------------
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()
