import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv("processed/feature_stock_data.csv")

# ==========================================
# Remove Missing Values
# ==========================================

df = df.dropna()

# ==========================================
# Features and Target
# ==========================================

X = df.drop(columns=["Target", "Date"])

y = df["Target"]

# ==========================================
# One-Hot Encoding
# ==========================================

X = pd.get_dummies(
    X,
    columns=[
        "Company",
        "Symbol",
        "Series"
    ]
)

# ==========================================
# Save Feature Names
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    X.columns.tolist(),
    "models/model_columns.pkl"
)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(X_test)

print("\nModel Performance")

print(
    "MAE :",
    mean_absolute_error(y_test, prediction)
)

print(
    "RMSE :",
    np.sqrt(mean_squared_error(y_test, prediction))
)

print(
    "R2 Score :",
    r2_score(y_test, prediction)
)

# ==========================================
# Save Model
# ==========================================


import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/best_model.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("\nModel saved successfully!")
print("Features used:", len(X.columns))