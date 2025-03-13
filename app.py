import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model and Preprocessor
model = joblib.load("optimized_readmission_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Define all expected columns with default values
EXPECTED_COLUMNS = [
    "num_lab_procedures",
    "num_procedures",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "time_in_hospital",
    "num_medications"
]

@app.route("/", methods=["GET"])
def home():
    return "Flask is running! Use /predict for predictions."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Ensure all expected columns exist
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = 0  # Assign default value if missing

        # Transform input using preprocessor
        df_processed = preprocessor.transform(df)

        # Make prediction
        prediction = model.predict(df_processed)

        return jsonify({"readmission_prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default Flask port
    app.run(host="0.0.0.0", port=port)
