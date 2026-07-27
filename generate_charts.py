import os
import pandas as pd
import matplotlib

# Prevent GUI warnings
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("processed/feature_stock_data.csv")

# ==========================
# Create Charts Folder
# ==========================

os.makedirs("static/charts", exist_ok=True)

# ==========================
# Chart 1
# Company-wise Average Close
# ==========================

company_avg = data.groupby("Company")["Close"].mean()

plt.figure(figsize=(10,5))
company_avg.sort_values().plot(kind="bar")

plt.title("Average Closing Price")
plt.xlabel("Company")
plt.ylabel("Average Close")

plt.tight_layout()

plt.savefig("static/charts/company_avg_close.png")
plt.close()

# ==========================
# Chart 2
# Trading Volume
# ==========================

company_volume = data.groupby("Company")["Volume"].mean()

plt.figure(figsize=(10,5))
company_volume.sort_values().plot(kind="bar", color="orange")

plt.title("Average Trading Volume")
plt.xlabel("Company")
plt.ylabel("Volume")

plt.tight_layout()

plt.savefig("static/charts/trading_volume.png")
plt.close()

# ==========================
# Chart 3
# Closing Price Distribution
# ==========================

plt.figure(figsize=(8,5))

plt.hist(
    data["Close"],
    bins=30,
    edgecolor="black"
)

plt.title("Closing Price Distribution")
plt.xlabel("Closing Price")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("static/charts/close_distribution.png")
plt.close()

print("Charts Generated Successfully!")