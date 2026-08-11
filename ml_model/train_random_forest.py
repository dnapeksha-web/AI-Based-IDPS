import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load encoded datasets
train_df = pd.read_csv("dataset/UNSW_NB15_training_encoded.csv")
test_df = pd.read_csv("dataset/UNSW_NB15_testing_encoded.csv")

# Features and labels
X_train = train_df.drop(columns=["attack_cat", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["attack_cat", "label"])
y_test = test_df["label"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")

# Train the model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nRandom Forest Accuracy: {accuracy:.4f}")

# Create models folder if it doesn't exist
os.makedirs("ml_model/models", exist_ok=True)

# Save the model
joblib.dump(model, "ml_model/models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully.")