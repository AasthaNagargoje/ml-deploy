import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Dataset (area, bedrooms, bathrooms, stories)
X = np.array([
    [1000, 2, 1, 1],
    [1500, 3, 2, 2],
    [1800, 3, 2, 2],
    [2400, 4, 3, 2],
    [3000, 4, 3, 3]
])

# Prices
y = np.array([3000000, 5000000, 5500000, 7000000, 9000000])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")