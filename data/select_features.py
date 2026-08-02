import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/network_traffic_cleaned.csv")

# Select input features
X = df[
    [
        "source_ip",
        "destination_ip",
        "source_port",
        "destination_port",
        "protocol",
        "packet_size"
    ]
]

# Select target label
y = df["label"]

print("===== Features (X) =====")
print(X.head())

print("\n===== Target (y) =====")
print(y.head())

print("\nFeature Columns:")
print(list(X.columns))

print("\nNumber of Features:", X.shape[1])
print("Number of Samples :", X.shape[0])