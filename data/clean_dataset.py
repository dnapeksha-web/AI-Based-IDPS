import pandas as pd

# Load the labeled dataset
df = pd.read_csv("data/network_traffic_labeled.csv")

print("Missing values before cleaning:")
print(df.isnull().sum())

# Replace missing port numbers with -1
df["source_port"] = df["source_port"].fillna(-1).astype(int)
df["destination_port"] = df["destination_port"].fillna(-1).astype(int)

# Save the cleaned dataset
df.to_csv("data/network_traffic_cleaned.csv", index=False)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset cleaned successfully.")
print(df.head())