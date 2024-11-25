from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load and preprocess the Excel data
df = pd.read_excel("ragweed.xlsx")

# Preprocess the data to extract features and labels
data = []

# Collect the data from each pattern (ignore November)
for i in range(901):  # Adjust the number of rows accordingly
    temp_data = df.iloc[i * 5:(i + 1) * 5].copy()
    
    # Convert month to numerical values and filter out November
    month_map = {'july': 7, 'august': 8, 'september': 9, 'october': 10}
    temp_data['month'] = temp_data['month'].str.lower().map(month_map)
    temp_data = temp_data[temp_data['month'].notna()]  # Exclude November rows
    
    data.append(temp_data)

# Stack the data into a single DataFrame
data_df = pd.concat(data).reset_index(drop=True)

# Extract features (month, temp, humidity, clouds, rain) and labels (start and end shiftings)
X = data_df[['month', 'temp (°C)', 'humidity (%)', 'clouds (%)', 'rain (mm)']].values
y = data_df[['start shifting (days)', 'end shifting (days)']].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Filter training data for start and end shifting
start_shift_train = X_train[X_train[:, 0] <= 8]  # Only July and August
start_shift_labels = y_train[X_train[:, 0] <= 8, 0]

end_shift_train = X_train[X_train[:, 0] <= 10]  # Use July to October
end_shift_labels = y_train[X_train[:, 0] <= 10, 1]

# Train two Decision Tree Regressors
start_shift_model = DecisionTreeRegressor(random_state=42)
end_shift_model = DecisionTreeRegressor(random_state=42)

start_shift_model.fit(start_shift_train[:, 1:], start_shift_labels)  # Use features excluding the month
end_shift_model.fit(end_shift_train[:, 1:], end_shift_labels)  # Use features excluding the month

# Function to predict shiftings based on given weather data
def predict_shiftings(weather_data):
    """
    Predict start and end shiftings based on the given weather data for the months July to October.
    :param weather_data: List of tuples [(month, temp, humidity, clouds, rain), ...]
    :return: Tuple (predicted_start_shift, predicted_end_shift)
    """
    start_shift_input = []
    end_shift_input = []

    for entry in weather_data:
        # If months are integers, use them directly; otherwise, map strings to integers
        if isinstance(entry[0], str):
            month_map = {'july': 7, 'august': 8, 'september': 9, 'october': 10}
            month = month_map.get(entry[0].lower())
        else:
            month = entry[0]
        
        if month is None:
            raise ValueError(f"Invalid month: {entry[0]}")
        
        temp, humidity, clouds, rain = entry[1:]
        
        # For start shifting, only consider July and August
        if month <= 8:
            start_shift_input.append([temp, humidity, clouds, rain])
        
        # For end shifting, use the last available month
        if month <= 10:
            end_shift_input = [[temp, humidity, clouds, rain]]  # Update to the most recent valid month
    
    # Predict start shifting if data for July or August is provided
    if start_shift_input:
        start_shift_pred = start_shift_model.predict(np.array(start_shift_input)).mean()
    else:
        start_shift_pred = 0  # No valid data for start shifting
    
    # Predict end shifting using the last available month
    if end_shift_input:
        end_shift_pred = end_shift_model.predict(np.array(end_shift_input))[0]
    else:
        end_shift_pred = 0  # No valid data for end shifting
    
    return {"start_shift": start_shift_pred, "end_shift": end_shift_pred}

@app.route("/")
def home():
    return "Welcome to the Allergy Prediction API! Use /predict to get predictions."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        weather_data = request.json.get("weather_data", [])
        if not weather_data:
            raise ValueError("No weather data provided.")
        result = predict_shiftings(weather_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
