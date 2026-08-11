import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load encoded datasets
train_df = pd.read_csv("dataset/UNSW_NB15_training_encoded.csv")
test_df = pd.read_csv("dataset/UNSW_NB15_testing_encoded.csv")

# Remove unnecessary column
train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])

# Features and labels
X_train = train_df.drop(columns=["attack_cat", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["attack_cat", "label"])
y_test = test_df["label"]

# Improved Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Training Improved Random Forest model...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n===== Improved Model =====\n")
print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall   : {recall_score(y_test, predictions):.4f}")
print(f"F1-Score : {f1_score(y_test, predictions):.4f}")

os.makedirs("ml_model/models", exist_ok=True)

joblib.dump(
    model,
    "ml_model/models/improved_random_forest.pkl"
)

print("\nImproved model saved successfully.")