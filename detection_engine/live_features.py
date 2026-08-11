from scapy.all import IP, TCP, UDP


# UNSW-NB15 feature columns expected by our model
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


def extract_live_features(packet):
    """
    Convert one Scapy packet into the 43 features
    expected by the trained Random Forest model.
    """

    # Basic defaults
    features = {column: 0 for column in FEATURE_COLUMNS}

    packet_size = len(packet)

    # We need an IP packet
    if IP not in packet:
        return features

    ip = packet[IP]

    # -----------------------------
    # Protocol
    # -----------------------------

    if TCP in packet:
        tcp = packet[TCP]

        features["proto"] = 113       # TCP encoding from our dataset
        features["spkts"] = 1
        features["dpkts"] = 0
        features["sbytes"] = packet_size
        features["dbytes"] = 0
        features["sttl"] = ip.ttl
        features["dttl"] = 0
        features["smean"] = packet_size
        features["dmean"] = 0

        # TCP window
        features["swin"] = tcp.window

        # TCP flags → approximate connection state
        flags = str(tcp.flags)

        if "S" in flags and "A" not in flags:
            features["state"] = 5
        elif "S" in flags and "A" in flags:
            features["state"] = 4
        elif "F" in flags:
            features["state"] = 2
        elif "R" in flags:
            features["state"] = 1
        else:
            features["state"] = 4

    elif UDP in packet:
        udp = packet[UDP]

        features["proto"] = 119      # UDP encoding from our dataset
        features["spkts"] = 1
        features["dpkts"] = 0
        features["sbytes"] = packet_size
        features["dbytes"] = 0
        features["sttl"] = ip.ttl
        features["dttl"] = 0
        features["smean"] = packet_size
        features["dmean"] = 0

        features["state"] = 4

    else:
        # Other IP protocols
        features["proto"] = 0
        features["spkts"] = 1
        features["sbytes"] = packet_size
        features["sttl"] = ip.ttl
        features["smean"] = packet_size
        features["state"] = 0

    # -----------------------------
    # Basic traffic statistics
    # -----------------------------

    features["dur"] = 0.0

    if features["dur"] > 0:
        features["rate"] = features["sbytes"] / features["dur"]
        features["sload"] = features["sbytes"] / features["dur"]

    features["ct_srv_src"] = 1
    features["ct_state_ttl"] = 1
    features["ct_dst_ltm"] = 1
    features["ct_src_dport_ltm"] = 1
    features["ct_dst_sport_ltm"] = 1
    features["ct_dst_src_ltm"] = 1
    features["ct_src_ltm"] = 1
    features["ct_srv_dst"] = 1

    # -----------------------------
    # Other available values
    # -----------------------------

    if TCP in packet:
        tcp = packet[TCP]

        features["stcpb"] = tcp.seq
        features["dwin"] = tcp.window

    # -----------------------------
    # Return only model features
    # -----------------------------

    return features


if __name__ == "__main__":
    from scapy.all import sniff

    print("Capturing 5 packets...\n")

    def process_packet(packet):
        features = extract_live_features(packet)

        print("--------------------------------")
        print("Live ML Features")
        print("--------------------------------")

        for column in FEATURE_COLUMNS:
            print(f"{column:22}: {features[column]}")

        print()

    sniff(
        count=5,
        prn=process_packet,
        store=False
    )

    print("Feature extraction completed.")