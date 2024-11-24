from flask import Flask, jsonify, request
import numpy as np
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Simulated dataset
data = [
    [7, 25, 60, 30, 10, -3.5, 7.0],  # July
    [8, 24, 65, 40, 20, -5.0, 5.0],  # August
    [9, 20, 70, 50, 30, 0.0, 6.5],   # September
    [10, 18, 75, 60, 40, 0.0, 7.5],  # October
]

# Train models
X = np.array([row[:5] for row in data])
y_start = np.array([row[5] for row in data])
y_end = np.array([row[6] for row in data])

start_shift_model = DecisionTreeRegressor(random_state=42)
end_shift_model = DecisionTreeRegressor(random_state=42)

start_shift_model.fit(X[:, 1:], y_start)
end_shift_model.fit(X[:, 1:], y_end)

# Prediction function
def predict_shiftings(weather_data):
    weather_data.sort(key=lambda x: x[0])
    start_inputs = [x[1:] for x in weather_data if x[0] in [7, 8]]
    end_inputs = [x[1:] for x in weather_data if x[0] <= 10]

    start_shift_pred = (
        float(np.mean(start_shift_model.predict(start_inputs))) if start_inputs else 0
    )
    end_shift_pred = (
        float(np.mean(end_shift_model.predict(end_inputs))) if end_inputs else 0
    )
    return {"start_shift": start_shift_pred, "end_shift": end_shift_pred}

# Define a route for the root
@app.route('/')
def home():
    return "Welcome to the Allergy Prediction API! Use /predict to get predictions."

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        weather_data = request.json['weather_data']
        result = predict_shiftings(weather_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
