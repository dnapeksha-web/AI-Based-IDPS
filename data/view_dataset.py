import pandas as pd

# Load the dataset
df = pd.read_csv("data/network_traffic.csv")

print("\n===== First 10 Rows =====")
print(df.head(10))

print("\n===== Dataset Information =====")
print(df.info())

print("\n===== Column Names =====")
print(df.columns.tolist())

print("\n===== Dataset Shape =====")
print(df.shape)