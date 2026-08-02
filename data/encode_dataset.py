import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the cleaned dataset
df = pd.read_csv("data/network_traffic_cleaned.csv")

# Create encoders
source_ip_encoder = LabelEncoder()
destination_ip_encoder = LabelEncoder()
protocol_encoder = LabelEncoder()
label_encoder = LabelEncoder()

# Encode categorical columns
df["source_ip"] = source_ip_encoder.fit_transform(df["source_ip"])
df["destination_ip"] = destination_ip_encoder.fit_transform(df["destination_ip"])
df["protocol"] = protocol_encoder.fit_transform(df["protocol"])
df["label"] = label_encoder.fit_transform(df["label"])

# Save the encoded dataset
df.to_csv("data/network_traffic_encoded.csv", index=False)

print("Dataset encoded successfully.\n")

print(df.head())

print("\nEncoded Protocol Values:")
for value, code in zip(protocol_encoder.classes_, protocol_encoder.transform(protocol_encoder.classes_)):
    print(f"{value} -> {code}")

print("\nEncoded Label Values:")
for value, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{value} -> {code}")