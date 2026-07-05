"""
generate_dataset.py
--------------------
Generates a synthetic historical weather dataset for flood prediction.

Real government/meteorological flood datasets (e.g. Kerala rainfall data,
Kaggle "Flood Prediction" datasets) use similar features. Since no dataset
file was supplied with this project, this script builds a realistic
synthetic dataset with the same structure, so the pipeline below can be
run end-to-end. If you have a real historical dataset (CSV), simply
replace data/flood_data.csv with it, keeping the same column names,
and re-run train_model.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000  # number of historical records (e.g. district-years)

# --- Features -----------------------------------------------------------
# Annual rainfall (mm) - realistic range for flood-prone regions
annual_rainfall = np.random.normal(2200, 650, N).clip(400, 4500)

# Seasonal (monsoon) rainfall (mm) - correlated with annual rainfall
seasonal_rainfall = (annual_rainfall * np.random.uniform(0.55, 0.75, N)
                      + np.random.normal(0, 120, N)).clip(200, 3200)

# Cloud visibility (km) - lower visibility during heavy rain/storm systems
cloud_visibility = (15 - (seasonal_rainfall / 3200) * 12
                     + np.random.normal(0, 1.5, N)).clip(0.5, 15)

# Extra supporting meteorological features
humidity = (55 + (seasonal_rainfall / 3200) * 35
            + np.random.normal(0, 5, N)).clip(20, 100)

temperature = np.random.normal(27, 4, N).clip(10, 42)

river_discharge = (500 + (seasonal_rainfall / 3200) * 4000
                    + np.random.normal(0, 300, N)).clip(50, 6000)

soil_moisture = (30 + (seasonal_rainfall / 3200) * 60
                  + np.random.normal(0, 6, N)).clip(5, 100)

# --- Target: flood occurrence -------------------------------------------
# Combine features into a flood-risk score, then threshold with noise
risk_score = (
    0.35 * (seasonal_rainfall / 3200) +
    0.20 * (annual_rainfall / 4500) +
    0.15 * (1 - cloud_visibility / 15) +
    0.15 * (humidity / 100) +
    0.10 * (river_discharge / 6000) +
    0.05 * (soil_moisture / 100)
)

# add noise and threshold
noise = np.random.normal(0, 0.06, N)
flood_prob = risk_score + noise
threshold = np.quantile(flood_prob, 0.55)  # keep classes reasonably balanced
flood = (flood_prob > threshold).astype(int)

df = pd.DataFrame({
    "annual_rainfall_mm": annual_rainfall.round(1),
    "seasonal_rainfall_mm": seasonal_rainfall.round(1),
    "cloud_visibility_km": cloud_visibility.round(2),
    "humidity_percent": humidity.round(1),
    "temperature_celsius": temperature.round(1),
    "river_discharge_cusecs": river_discharge.round(1),
    "soil_moisture_percent": soil_moisture.round(1),
    "flood": flood
})

df.to_csv("/home/claude/flood_prediction_project/data/flood_data.csv", index=False)
print("Dataset generated:", df.shape)
print(df["flood"].value_counts())
print(df.head())
