import pandas as pd

# Load the datasets
train_df = pd.read_csv("dataset/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("dataset/UNSW_NB15_testing-set.csv")

# Features (remove attack_cat and label)
X_train = train_df.drop(columns=["attack_cat", "label"])
X_test = test_df.drop(columns=["attack_cat", "label"])

# Target
y_train = train_df["label"]
y_test = test_df["label"]

print("===== Training Data =====")
print("Features Shape :", X_train.shape)
print("Labels Shape   :", y_train.shape)

print("\n===== Testing Data =====")
print("Features Shape :", X_test.shape)
print("Labels Shape   :", y_test.shape)

print("\nTarget Classes:")
print(y_train.value_counts())