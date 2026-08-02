import pandas as pd
from sklearn.model_selection import train_test_split

# Load the encoded dataset
df = pd.read_csv("data/network_traffic_encoded.csv")

# Features (X)
X = df.drop("label", axis=1)

# Target (y)
y = df["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dataset split successfully.\n")

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print("\nTraining Features Shape :", X_train.shape)
print("Testing Features Shape  :", X_test.shape)

print("\nTraining Labels Shape   :", y_train.shape)
print("Testing Labels Shape    :", y_test.shape)