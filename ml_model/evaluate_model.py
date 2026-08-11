import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load encoded testing dataset
test_df = pd.read_csv("dataset/UNSW_NB15_testing_encoded.csv")

# Features and labels
X_test = test_df.drop(columns=["attack_cat", "label"])
y_test = test_df["label"]

# Load trained model
model = joblib.load("ml_model/models/random_forest_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
print("\n===== Model Evaluation =====\n")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))