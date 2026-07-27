import pandas as pd
import joblib
from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
# ==========================
# Create Flask App
# ==========================
app = Flask(__name__)

# ==========================
# Load Model
# ==========================
model = joblib.load("models/best_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

model_columns = joblib.load("models/model_columns.pkl")

print("=" * 50)
print("Total Model Columns:", len(model_columns))
print(model_columns)
print("=" * 50)

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("processed/feature_stock_data.csv")

# ==========================
# Company List
# ==========================
companies = sorted(data["Company"].unique())


# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# About Page
# ==========================
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/performance")
def performance():
    return render_template("performance.html")


# ==========================
# Prediction Page
# ==========================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    company = None
    symbol = None
    latest_close = None
    difference = None
    trend = None
    recent_data = []

    if request.method == "POST":

        company = request.form["company"]

        # Get latest record for selected company
        latest = data[data["Company"] == company].iloc[-1]

        symbol = latest["Symbol"]

        # Create input dictionary
        input_data = {}

        # Set all features to 0
        for col in model_columns:
            input_data[col] = 0

        # Numerical columns
        numeric_columns = [
            "Prev Close",
            "Open",
            "High",
            "Low",
            "Last",
            "Close",
            "VWAP",
            "Volume",
            "Turnover",
            "Trades",
            "Deliverable Volume",
            "%Deliverble",
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Lag_5",
            "MA_5",
            "MA_10",
            "MA_20",
            "EMA_10",
            "EMA_20",
            "Daily_Return",
            "Price_Range",
            "Open_Close_Diff",
            "Volatility"
        ]

        for col in numeric_columns:
            input_data[col] = latest[col]

        # Company
        company_col = f"Company_{company}"
        if company_col in input_data:
            input_data[company_col] = 1

        # Symbol
        symbol_col = f"Symbol_{latest['Symbol']}"
        if symbol_col in input_data:
            input_data[symbol_col] = 1

        # Series
        series_col = f"Series_{latest['Series']}"
        if series_col in input_data:
            input_data[series_col] = 1

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Prediction
        prediction = model.predict(input_df)[0]

        latest_close = latest["Close"]

        # ==========================
# Generate Stock Price Chart
# ==========================

        company_data = data[data["Company"] == company].tail(30)

        recent_data = company_data[["Date", "Open", "High", "Low", "Close"]]
        recent_data = recent_data.tail(10)
        recent_data = recent_data.to_dict(orient="records")

        plt.figure(figsize=(8,4))
        plt.plot(
            company_data["Close"],
            marker="o"
        )

        plt.title(f"{company} Closing Price")
        plt.xlabel("Last 30 Records")
        plt.ylabel("Closing Price")

        chart_path = os.path.join(
            "static",
            "charts",
            "stock_chart.png"
        )

        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        difference = prediction - latest_close

        if difference > 0:
            trend = "📈 Price Expected to Increase"
        elif difference < 0:
            trend = "📉 Price Expected to Decrease"
        else:
            trend = "➡️ Price Expected to Remain Stable"

    return render_template(
        "predict.html",
        companies=companies,
        prediction=prediction,
        company=company,
        symbol=symbol,
        latest_close=latest_close,
        difference=difference,
        trend=trend,
        chart="charts/stock_chart.png",
        recent_data=recent_data
)
# ==========================
# Run Application
# ==========================
if __name__ == "__main__":
    app.run(debug=True)