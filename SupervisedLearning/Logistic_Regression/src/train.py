import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Load cleaned data
data_path = os.path.join("data", "processed", "cleaned_churn_data.csv")
df = pd.read_csv(data_path)

# 2. Separate features (X) and target (y)
X = df.drop(columns=["churn"])
y = df["churn"]

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Initialize and train Logistic Regression model
# Balance the minority churn class so the model learns to identify more churners.
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 6. Evaluate Model Metrics
print("--- Model Performance Metrics ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# 7. Save Model Artifact
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "model.pkl")
joblib.dump(model, model_path)
print(f"\nTrained model successfully saved to: {model_path}")