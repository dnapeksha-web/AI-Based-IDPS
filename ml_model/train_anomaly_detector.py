import os
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_DATA = r"dataset\UNSW_NB15_training_encoded.csv"

MODEL_DIR = r"ml_model\models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "improved_anomaly_detector.pkl"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports"
]


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(TRAINING_DATA)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. SELECT NORMAL TRAFFIC
# ============================================================

print("\nSelecting normal traffic...")

normal_df = df[df["label"] == 0].copy()

print("Normal samples:", len(normal_df))


# ============================================================
# 3. SELECT FEATURES
# ============================================================

print("\nSelecting security features...")

X = normal_df[FEATURE_COLUMNS].copy()

print("Feature count:", len(FEATURE_COLUMNS))
print("Feature shape:", X.shape)


# ============================================================
# 4. CLEAN INVALID VALUES
# ============================================================

print("\nCleaning feature values...")

X = X.replace([float("inf"), float("-inf")], 0)

X = X.fillna(0)


# ============================================================
# 5. ROBUST SCALING
# ============================================================

print("\nApplying RobustScaler...")

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed.")


# ============================================================
# 6. TRAIN ISOLATION FOREST
# ============================================================

print("\nTraining improved Isolation Forest...")

model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination=0.02,
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)

print("Isolation Forest training completed.")


# ============================================================
# 7. TRAINING DATA CHECK
# ============================================================

predictions = model.predict(X_scaled)

normal_predictions = (predictions == 1).sum()
anomaly_predictions = (predictions == -1).sum()

print("\n================================")
print("TRAINING DATA CHECK")
print("================================")

print("Normal predictions :", normal_predictions)
print("Anomalies detected  :", anomaly_predictions)

anomaly_rate = (
    anomaly_predictions / len(predictions)
) * 100

print(f"Training anomaly rate: {anomaly_rate:.2f}%")


# ============================================================
# 8. SAVE MODEL + SCALER
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

model_package = {
    "model": model,
    "scaler": scaler,
    "features": FEATURE_COLUMNS
}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\nImproved anomaly detector saved successfully:")
print(MODEL_PATH)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n================================")
print("ANOMALY DETECTOR TRAINING DONE")
print("================================")

print("Training samples :", len(X))
print("Features         :", len(FEATURE_COLUMNS))
print("Trees            :", 300)
print("Contamination    : 0.02")
print("Scaler           : RobustScaler")