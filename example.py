import joblib

columns = joblib.load("models/model_columns.pkl")

print("Total Columns:", len(columns))
print(columns)