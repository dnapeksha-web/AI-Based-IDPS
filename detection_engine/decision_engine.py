def make_decision(prediction, confidence):
    """
    Convert ML prediction and confidence
    into a security decision.
    """

    if prediction == 0:
        return {
            "threat_level": "NORMAL",
            "action": "ALLOW"
        }

    if confidence < 70:
        return {
            "threat_level": "SUSPICIOUS",
            "action": "ALERT"
        }

    return {
        "threat_level": "MALICIOUS",
        "action": "BLOCK"
    }


if __name__ == "__main__":

    test_cases = [
        (0, 95),
        (1, 60),
        (1, 85)
    ]

    for prediction, confidence in test_cases:

        result = make_decision(
            prediction,
            confidence
        )

        print("--------------------------------")
        print("Prediction   :", prediction)
        print("Confidence   :", confidence)
        print("Threat Level :", result["threat_level"])
        print("Action       :", result["action"])