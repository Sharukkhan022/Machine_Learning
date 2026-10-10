import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/raw/Bank Customer Churn Prediction.csv')

# Drop identifier column
df.drop(columns=['customer_id'], inplace=True)

# Encode gender as binary (Female: 0, Male: 1)
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# One-hot encode country, dropping first category to avoid multi-collinearity
df = pd.get_dummies(df, columns=['country'], drop_first=True, dtype=int)

# print(df.columns)

# 4. Separate features (X) and target (y)
X = df.drop(columns=["churn"])
y = df["churn"]

# 5. Apply StandardScaler on numerical features
num_cols = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 6. Recombine and save processed dataset + fitted scaler
df_processed = pd.concat([X, y], axis=1)

os.makedirs(os.path.join("data", "processed"), exist_ok=True)
os.makedirs("models", exist_ok=True)

df_processed.to_csv(os.path.join("data", "processed", "cleaned_churn_data.csv"), index=False)
joblib.dump(scaler, os.path.join("models", "scaler.pkl"))

print("Data cleaning complete. Saved processed CSV and scaler object.")