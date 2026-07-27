import joblib

model = joblib.load("models/best_model.pkl")

print("Number of features:", len(model.feature_names_in_))
print()

for col in model.feature_names_in_[:50]:
    print(col)