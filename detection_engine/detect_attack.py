from scapy.all import sniff, IP, IPv6
import joblib
import pandas as pd

from live_features import extract_live_features, FEATURE_COLUMNS
from decision_engine import make_decision
from prevention import prevent_ip


# -----------------------------------
# Load trained model
# -----------------------------------

MODEL_PATH = "ml_model/models/improved_random_forest.pkl"

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")
print("Decision Engine loaded successfully.")
print("Prevention system loaded successfully.")
print("Starting live traffic analysis...")
print("Capturing 5 packets...\n")


# -----------------------------------
# Get source IP
# -----------------------------------

def get_source_ip(packet):

    if IP in packet:
        return packet[IP].src

    if IPv6 in packet:
        return packet[IPv6].src

    return "UNKNOWN"


# -----------------------------------
# Process packet
# -----------------------------------

def process_packet(packet):

    # Extract features
    features = extract_live_features(packet)

    # Convert features into DataFrame
    feature_data = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # ML prediction
    prediction = model.predict(feature_data)[0]

    # Prediction confidence
    probabilities = model.predict_proba(feature_data)[0]
    confidence = max(probabilities) * 100

    # Decision Engine
    decision = make_decision(
        prediction,
        confidence
    )

    # Source IP
    source_ip = get_source_ip(packet)

    # -----------------------------------
    # Display analysis
    # -----------------------------------

    print("--------------------------------")
    print("LIVE SECURITY ANALYSIS")
    print("--------------------------------")

    print("Source IP    :", source_ip)
    print("Packet Size  :", len(packet))
    print("Prediction   :", prediction)
    print(f"Confidence   : {confidence:.2f}%")
    print("Threat Level :", decision["threat_level"])
    print("Action       :", decision["action"])

    # -----------------------------------
    # Prevention
    # -----------------------------------

    if decision["action"] == "BLOCK":

        result = prevent_ip(source_ip)

        print("Prevention   :", result["status"])
        print("Blocked IP   :", result["ip"])

    elif decision["action"] == "ALERT":

        print("Alert        : Suspicious traffic detected")

    else:

        print("Result       : Traffic allowed")

    print()


# -----------------------------------
# Capture packets
# -----------------------------------

sniff(
    count=5,
    prn=process_packet,
    store=False
)

print("Live security analysis completed.")