from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask import Flask
import joblib

# Create a Flask application instance
app = Flask(__name__)

# Load the pre-trained Linear Regression model
linear_reg_model_loaded = joblib.load('linear_regression_model.joblib')
print("Linear Regression model loaded successfully.")

# Load the pre-trained Random Forest Regressor model
random_forest_model_loaded = joblib.load('random_forest_regressor_model.joblib')
print("Random Forest Regressor model loaded successfully.")

app = Flask(__name__)

# -------------------- Load Models --------------------
try:
    linear_reg_model = joblib.load("linear_regression_model.joblib")
    print("✅ Linear Regression model loaded successfully.")
except Exception as e:
    print("❌ Error loading Linear Regression model:", e)

try:
    random_forest_model = joblib.load("random_forest_regressor_model.joblib")
    print("✅ Random Forest model loaded successfully.")
except Exception as e:
    print("❌ Error loading Random Forest model:", e)

# -------------------- Prediction Route --------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Ensure request contains JSON
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400

    data = request.get_json()

    required_features = [
        "hours_studied",
        "sleep_hours",
        "attendance_percent",
        "previous_scores"
    ]

    # Check missing features
    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing
        }), 400

    # Convert inputs to float safely
    try:
        input_data = {
            "hours_studied": float(data["hours_studied"]),
            "sleep_hours": float(data["sleep_hours"]),
            "attendance_percent": float(data["attendance_percent"]),
            "previous_scores": float(data["previous_scores"])
        }
    except ValueError:
        return jsonify({
            "error": "All input values must be numeric"
        }), 400

    # Convert to DataFrame (important for sklearn)
    input_df = pd.DataFrame([input_data])

    try:
        # Predictions
        lr_pred = linear_reg_model.predict(input_df)[0]
        rf_pred = random_forest_model.predict(input_df)[0]
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

    return jsonify({
        "linear_regression_prediction": round(float(lr_pred), 2),
        "random_forest_prediction": round(float(rf_pred), 2)
    }), 200


# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
