import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("processed/feature_stock_data.csv")

# ==========================
# Remove Missing Values
# ==========================
data = data.dropna()

# ==========================
# Convert Categorical Columns
# ==========================
data = pd.get_dummies(
    data,
    columns=["Company", "Series", "Symbol"]
)

# ==========================
# Features and Target
# ==========================
X = data.drop(columns=["Target", "Date"])
y = data["Target"]

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Random Forest Model
# ==========================
forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train, y_train)

# ==========================
# Save Model
# ==========================
joblib.dump(forest, "models/best_model.pkl")

# ==========================
# Save Feature Names
# ==========================
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model Saved Successfully!")