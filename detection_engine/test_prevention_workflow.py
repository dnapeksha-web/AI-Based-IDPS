from decision_engine import make_decision
from prevention import prevent_ip
from ip_blacklist import is_blocked


def test_workflow():

    # Simulated ML result
    prediction = 1
    confidence = 90

    # Test source IP
    source_ip = "192.168.31.200"

    print("===== PREVENTION WORKFLOW TEST =====")
    print()

    # Step 1: Decision Engine
    decision = make_decision(
        prediction,
        confidence
    )

    print("Prediction   :", prediction)
    print("Confidence   :", confidence)
    print("Threat Level :", decision["threat_level"])
    print("Action       :", decision["action"])
    print()

    # Step 2: Prevention
    if decision["action"] == "BLOCK":

        result = prevent_ip(source_ip)

        print("Prevention Status :", result["status"])
        print("Blocked IP        :", result["ip"])

    # Step 3: Verify blacklist
    print()
    print("Blacklist Check   :", is_blocked(source_ip))

    print()
    print("===== TEST COMPLETED =====")


if __name__ == "__main__":
    test_workflow()