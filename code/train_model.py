import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import joblib
import os

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/flood_data.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==============================
# Handle Missing Values
# ==============================

df.fillna(df.mean(numeric_only=True), inplace=True)

# ==============================
# Data Visualization
# ==============================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

df.hist(figsize=(12,10))
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# ==============================
# Feature Selection
# ==============================
possible_targets = ["Flood", "flood", "Flood_Chance", "FloodChance", "Target"]

TARGET_COLUMN = None

for col in possible_targets:
    if col in df.columns:
        TARGET_COLUMN = col
        break

if TARGET_COLUMN is None:
    TARGET_COLUMN = df.columns[-1]

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# Models
# ==============================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

best_accuracy = 0
best_model = None
best_name = ""

# ==============================
# Training
# ==============================

for name, model in models.items():

    print("\n====================================")
    print(name)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("Accuracy:", accuracy)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# ==============================
# Save Best Model
# ==============================

os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/best_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n====================================")
print("Best Model :", best_name)
print("Accuracy :", best_accuracy)
print("Model Saved Successfully!")