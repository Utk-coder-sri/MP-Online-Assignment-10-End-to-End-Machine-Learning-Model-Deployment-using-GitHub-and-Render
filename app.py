# ============================================
# Assignment 10
# Task 3 : API Development using Flask
# ============================================

# Import Libraries
from flask import Flask, request, jsonify
import joblib
import pandas as pd

print("\n==============================")
print("Task 3 : API Development")
print("==============================")

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Heart Disease Detected"
    else:
        result = "Heart Disease Not Detected"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
