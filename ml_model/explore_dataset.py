import pandas as pd

# Load training dataset
train_df = pd.read_csv("dataset/UNSW_NB15_training-set.csv")

# Load testing dataset
test_df = pd.read_csv("dataset/UNSW_NB15_testing-set.csv")

print("\n========== TRAINING DATASET ==========\n")

print("Shape:")
print(train_df.shape)

print("\nFirst 5 Rows:")
print(train_df.head())

print("\nColumn Names:")
print(train_df.columns.tolist())

print("\nData Types:")
print(train_df.dtypes)

print("\n========== TESTING DATASET ==========\n")

print("Shape:")
print(test_df.shape)