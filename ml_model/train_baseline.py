import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load encoded datasets
train_df = pd.read_csv("dataset/UNSW_NB15_training_encoded.csv")
test_df = pd.read_csv("dataset/UNSW_NB15_testing_encoded.csv")

# Features and labels
X_train = train_df.drop(columns=["attack_cat", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["attack_cat", "label"])
y_test = test_df["label"]

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
print("Training Decision Tree model...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nBaseline Accuracy: {accuracy:.4f}")

# Create models folder
os.makedirs("ml_model/models", exist_ok=True)

# Save model
joblib.dump(model, "ml_model/models/baseline_decision_tree.pkl")

print("\nBaseline model saved successfully.")