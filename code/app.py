from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Read dataset
df = pd.read_csv("data/flood_data.csv")

# Automatically detect target column
target_columns = ["Flood", "flood", "Flood_Chance", "FloodChance", "Target"]

target = None

for col in target_columns:
    if col in df.columns:
        target = col
        break

if target is None:
    target = df.columns[-1]

# Input features only
FEATURES = [col for col in df.columns if col != target]


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model-comparison")
def model_comparison():
    return render_template("model_comparison.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = []

    for feature in FEATURES:
        values.append(float(request.form[feature]))

    input_df = pd.DataFrame([values], columns=FEATURES)

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]

    if prediction == 1:
        result = "⚠ Flood Chance Detected"
    else:
        result = "✅ No Flood Chance"

    return render_template(
        "result.html",
        prediction=result,
        values=input_df.iloc[0].to_dict()
    )


if __name__ == "__main__":
    app.run(debug=True)