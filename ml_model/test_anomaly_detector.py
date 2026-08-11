import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"ml_model\models\improved_anomaly_detector.pkl"
TEST_DATA = r"dataset\UNSW_NB15_testing_encoded.csv"

OUTPUT_PATH = r"ml_model\improved_anomaly_test_results.csv"


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("Loading improved anomaly detector...")

package = joblib.load(MODEL_PATH)

model = package["model"]
scaler = package["scaler"]
feature_columns = package["features"]

print("Improved anomaly detector loaded successfully.")
print("Expected features:", len(feature_columns))


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading testing dataset...")

df = pd.read_csv(TEST_DATA)

print("Testing dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

print("\nSelecting model features...")

X = df[feature_columns].copy()

X = X.replace([float("inf"), float("-inf")], 0)
X = X.fillna(0)

print("Feature shape:", X.shape)


# ============================================================
# 4. SCALE TEST DATA
# ============================================================

print("\nApplying saved scaler...")

X_scaled = scaler.transform(X)

print("Scaling completed.")


# ============================================================
# 5. ANOMALY PREDICTION
# ============================================================

print("\nRunning anomaly detection...")

predictions = model.predict(X_scaled)

df["anomaly_prediction"] = predictions

df["anomaly_status"] = df["anomaly_prediction"].map(
    {
        1: "NORMAL",
        -1: "ANOMALY"
    }
)


# ============================================================
# 6. OVERALL RESULTS
# ============================================================

print("\n================================")
print("OVERALL RESULTS")
print("================================")

print(
    df["anomaly_status"].value_counts()
)


# ============================================================
# 7. NORMAL TRAFFIC
# ============================================================

normal_df = df[df["label"] == 0]

normal_total = len(normal_df)

normal_anomalies = (
    normal_df["anomaly_prediction"] == -1
).sum()

normal_correct = (
    normal_df["anomaly_prediction"] == 1
).sum()


false_positive_rate = (
    normal_anomalies / normal_total
) * 100


normal_detection_rate = (
    normal_correct / normal_total
) * 100


print("\n================================")
print("NORMAL TRAFFIC")
print("================================")

print("Total normal traffic :", normal_total)
print("Correctly normal     :", normal_correct)
print("False anomalies      :", normal_anomalies)

print(
    f"Normal detection rate: {normal_detection_rate:.2f}%"
)

print(
    f"False-positive rate   : {false_positive_rate:.2f}%"
)


# ============================================================
# 8. ATTACK TRAFFIC
# ============================================================

attack_df = df[df["label"] == 1]

attack_total = len(attack_df)

attack_anomalies = (
    attack_df["anomaly_prediction"] == -1
).sum()

attack_missed = (
    attack_df["anomaly_prediction"] == 1
).sum()


attack_detection_rate = (
    attack_anomalies / attack_total
) * 100


print("\n================================")
print("ATTACK TRAFFIC")
print("================================")

print("Total attacks       :", attack_total)
print("Detected anomalies  :", attack_anomalies)
print("Missed attacks      :", attack_missed)

print(
    f"Anomaly detection rate: {attack_detection_rate:.2f}%"
)


# ============================================================
# 9. ZERO-DAY SUPPORT
# ============================================================

print("\n================================")
print("ZERO-DAY / UNKNOWN THREAT SUPPORT")
print("================================")

print(
    "Anomaly detection is used to identify "
    "traffic that differs from learned normal behavior."
)

print(
    "Traffic classified as anomalous while not "
    "being confidently identified by the supervised "
    "model can be treated as potentially unknown activity."
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_PATH)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n================================")
print("IMPROVED ANOMALY TEST COMPLETED")
print("================================")

print(
    f"False-positive rate    : {false_positive_rate:.2f}%"
)

print(
    f"Attack anomaly rate    : {attack_detection_rate:.2f}%"
)