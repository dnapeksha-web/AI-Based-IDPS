import pandas as pd

# Load the dataset
df = pd.read_csv("data/network_traffic.csv")

# Add a label column
df["label"] = "Normal"

# Save the updated dataset
df.to_csv("data/network_traffic_labeled.csv", index=False)

print("Dataset labeled successfully.")
print(df.head())