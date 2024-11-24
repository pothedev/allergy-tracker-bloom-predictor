import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Simulated dataset
data = [
    [7, 25, 60, 30, 10, -3.5, 7.0],  # July
    [8, 24, 65, 40, 20, -5.0, 5.0],  # August
    [9, 20, 70, 50, 30, 0.0, 6.5],   # September
    [10, 18, 75, 60, 40, 0.0, 7.5],  # October
]

# Split features (X) and labels (y)
X = np.array([row[:5] for row in data])  # [month, temp, humidity, clouds, rain]
y_start = np.array([row[5] for row in data])  # Start shifting labels
y_end = np.array([row[6] for row in data])    # End shifting labels

# Train-test split
X_train, X_test, y_start_train, y_start_test, y_end_train, y_end_test = train_test_split(
    X, y_start, y_end, test_size=0.5, random_state=42
)

# Train models
start_shift_model = DecisionTreeRegressor(random_state=42)
end_shift_model = DecisionTreeRegressor(random_state=42)

start_shift_model.fit(X_train[:, 1:], y_start_train)  # Exclude month column for features
end_shift_model.fit(X_train[:, 1:], y_end_train)

# Prediction function
def predict_shiftings(weather_data):
    """
    Predict start and end shiftings based on given weather data.
    - For start shifting: Use July and earlier months up to the current month (exclusive).
    - For end shifting: Use all months up to the current month (exclusive).
    """
    start_inputs = []
    end_inputs = []

    # Sort weather data by month (ascending)
    weather_data.sort(key=lambda x: x[0])

    # Dynamically filter months for start and end shifting
    for month, temp, humidity, clouds, rain in weather_data:
        if month == 7 or month == 8:  # July and August for start shifting
            start_inputs.append([temp, humidity, clouds, rain])
        if month >= 7 and month <= 10:  # All months up to October for end shifting
            end_inputs.append([temp, humidity, clouds, rain])
    
    # Start shifting prediction
    start_shift_pred = (
        np.mean(start_shift_model.predict(np.array(start_inputs))) if start_inputs else 0
    )
    
    # End shifting prediction
    end_shift_pred = (
        np.mean(end_shift_model.predict(np.array(end_inputs))) if end_inputs else 0
    )
    
    return start_shift_pred, end_shift_pred

# Testing the function
weather_data = [
    [7, 25, 60, 30, 10],  # July
    [8, 24, 65, 40, 20],  # August
    [9, 20, 70, 50, 30]   # September
]

predicted_start, predicted_end = predict_shiftings(weather_data)
print(f"Predicted Start Shifting: {predicted_start:.2f} days")
print(f"Predicted End Shifting: {predicted_end:.2f} days")

# Calculate MSE for testing set
start_shift_pred = start_shift_model.predict(X_test[:, 1:])
end_shift_pred = end_shift_model.predict(X_test[:, 1:])

start_mse = mean_squared_error(y_start_test, start_shift_pred)
end_mse = mean_squared_error(y_end_test, end_shift_pred)

print(f"Start Shifting MSE: {start_mse:.2f}")
print(f"End Shifting MSE: {end_mse:.2f}")

# Root Mean Squared Error (RMSE)
start_rmse = np.sqrt(start_mse)
end_rmse = np.sqrt(end_mse)

print(f"Start Shifting RMSE: {start_rmse:.2f} days")
print(f"End Shifting RMSE: {end_rmse:.2f} days")
