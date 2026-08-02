import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_df = pd.read_csv("dataset/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("dataset/UNSW_NB15_testing-set.csv")

# Create LabelEncoders
proto_encoder = LabelEncoder()
service_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Combine train and test for consistent encoding
combined = pd.concat([train_df, test_df], ignore_index=True)

proto_encoder.fit(combined["proto"])
service_encoder.fit(combined["service"])
state_encoder.fit(combined["state"])

# Encode training data
train_df["proto"] = proto_encoder.transform(train_df["proto"])
train_df["service"] = service_encoder.transform(train_df["service"])
train_df["state"] = state_encoder.transform(train_df["state"])

# Encode testing data
test_df["proto"] = proto_encoder.transform(test_df["proto"])
test_df["service"] = service_encoder.transform(test_df["service"])
test_df["state"] = state_encoder.transform(test_df["state"])

# Save encoded datasets
train_df.to_csv("dataset/UNSW_NB15_training_encoded.csv", index=False)
test_df.to_csv("dataset/UNSW_NB15_testing_encoded.csv", index=False)

print("Datasets encoded successfully.\n")

print("Encoded columns:")
print(train_df[["proto", "service", "state"]].head())

print("\nNumber of Protocols :", len(proto_encoder.classes_))
print("Number of Services  :", len(service_encoder.classes_))
print("Number of States    :", len(state_encoder.classes_))