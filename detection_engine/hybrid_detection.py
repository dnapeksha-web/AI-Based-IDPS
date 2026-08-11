import joblib
import pandas as pd


# ============================================================
# MODEL PATHS
# ============================================================

RF_MODEL_PATH = (
    r"ml_model\models\improved_random_forest.pkl"
)

ANOMALY_MODEL_PATH = (
    r"ml_model\models\improved_anomaly_detector.pkl"
)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("Loading Random Forest model...")

rf_model = joblib.load(RF_MODEL_PATH)

print("Random Forest loaded successfully.")


# ============================================================
# LOAD ANOMALY DETECTOR
# ============================================================

print("Loading Isolation Forest...")

anomaly_package = joblib.load(
    ANOMALY_MODEL_PATH
)

anomaly_model = anomaly_package["model"]
anomaly_scaler = anomaly_package["scaler"]
anomaly_features = anomaly_package["features"]

print("Isolation Forest loaded successfully.")


# ============================================================
# MODEL FEATURES
# ============================================================

RF_FEATURES = list(
    rf_model.feature_names_in_
)


# ============================================================
# HYBRID DETECTION FUNCTION
# ============================================================

def hybrid_detect(features):
    """
    Perform hybrid detection using:

    1. Random Forest
    2. Isolation Forest
    """

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    if isinstance(features, dict):

        data = pd.DataFrame(
            [features]
        )

    elif isinstance(features, pd.DataFrame):

        data = features.copy()

    else:

        raise TypeError(
            "Features must be a dictionary or DataFrame."
        )


    # --------------------------------------------------------
    # Ensure required features exist
    # --------------------------------------------------------

    missing_rf = [
        feature
        for feature in RF_FEATURES
        if feature not in data.columns
    ]

    missing_anomaly = [
        feature
        for feature in anomaly_features
        if feature not in data.columns
    ]


    if missing_rf:

        raise ValueError(
            f"Missing Random Forest features: {missing_rf}"
        )


    if missing_anomaly:

        raise ValueError(
            f"Missing anomaly features: {missing_anomaly}"
        )


    # --------------------------------------------------------
    # Random Forest prediction
    # --------------------------------------------------------

    rf_input = data[RF_FEATURES]

    rf_prediction = int(
        rf_model.predict(rf_input)[0]
    )


    rf_probabilities = rf_model.predict_proba(
        rf_input
    )[0]

    rf_confidence = float(
        max(rf_probabilities) * 100
    )


    # --------------------------------------------------------
    # Isolation Forest prediction
    # --------------------------------------------------------

    anomaly_input = data[anomaly_features].copy()

    anomaly_input = anomaly_input.replace(
        [float("inf"), float("-inf")],
        0
    )

    anomaly_input = anomaly_input.fillna(0)


    anomaly_scaled = anomaly_scaler.transform(
        anomaly_input
    )


    anomaly_prediction = int(
        anomaly_model.predict(
            anomaly_scaled
        )[0]
    )


    is_anomaly = (
        anomaly_prediction == -1
    )


    # --------------------------------------------------------
    # HYBRID DECISION
    # --------------------------------------------------------

    if rf_prediction == 1:

        threat_level = "MALICIOUS"
        action = "BLOCK"

    elif is_anomaly:

        threat_level = "SUSPICIOUS"
        action = "ALERT"

    else:

        threat_level = "NORMAL"
        action = "ALLOW"


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "rf_prediction":
            rf_prediction,

        "rf_confidence":
            round(rf_confidence, 2),

        "anomaly_prediction":
            anomaly_prediction,

        "is_anomaly":
            is_anomaly,

        "threat_level":
            threat_level,

        "action":
            action
    }


    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n================================")
    print("HYBRID DETECTION ENGINE TEST")
    print("================================")


    # --------------------------------------------------------
    # Load one testing sample
    # --------------------------------------------------------

    test_path = (
        r"dataset\UNSW_NB15_testing_encoded.csv"
    )

    df = pd.read_csv(
        test_path
    )


    sample = df.iloc[0]


    features = {
        feature: sample[feature]
        for feature in RF_FEATURES
    }


    # --------------------------------------------------------
    # Run hybrid detection
    # --------------------------------------------------------

    result = hybrid_detect(
        features
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n===== HYBRID SECURITY ANALYSIS =====")

    print(
        "Random Forest Prediction :",
        result["rf_prediction"]
    )

    print(
        "RF Confidence            :",
        f'{result["rf_confidence"]:.2f}%'
    )

    print(
        "Anomaly Prediction       :",
        result["anomaly_prediction"]
    )

    print(
        "Anomalous Traffic       :",
        result["is_anomaly"]
    )

    print(
        "Threat Level             :",
        result["threat_level"]
    )

    print(
        "Action                   :",
        result["action"]
    )

    print(
        "\nHybrid detection test completed."
    )