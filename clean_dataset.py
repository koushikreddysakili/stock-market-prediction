import pandas as pd

# Load dataset
df = pd.read_csv("processed/feature_stock_data.csv")

print("Before Cleaning")
print(df["Company"].unique())

# Remove incorrect company names
df["Company"] = df["Company"].replace({
    "RELIANCE (1)": "RELIANCE",
    "TCS-selected-columns": "TCS"
})

# Remove duplicate rows
df = df.drop_duplicates()

print("\nAfter Cleaning")
print(sorted(df["Company"].unique()))

# Save cleaned dataset
df.to_csv("processed/feature_stock_data.csv", index=False)

print("\nDataset cleaned successfully!")