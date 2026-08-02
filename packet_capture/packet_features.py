from scapy.all import sniff, IP, IPv6, TCP, UDP


def display_packet_features(packet):

    print("\n--------------------------------")

    # Packet Size
    print(f"Packet Size      : {len(packet)} bytes")

    # Source IP and Destination IP
    if packet.haslayer(IP):
        source_ip = packet[IP].src
        destination_ip = packet[IP].dst

    elif packet.haslayer(IPv6):
        source_ip = packet[IPv6].src
        destination_ip = packet[IPv6].dst

    else:
        source_ip = "N/A"
        destination_ip = "N/A"

    print(f"Source IP        : {source_ip}")
    print(f"Destination IP   : {destination_ip}")

    # TCP Protocol
    if packet.haslayer(TCP):
        source_port = packet[TCP].sport
        destination_port = packet[TCP].dport
        protocol = "TCP"

    # UDP Protocol
    elif packet.haslayer(UDP):
        source_port = packet[UDP].sport
        destination_port = packet[UDP].dport
        protocol = "UDP"

    # Other Protocols
    else:
        source_port = "N/A"
        destination_port = "N/A"
        protocol = "OTHER"

    print(f"Source Port      : {source_port}")
    print(f"Destination Port : {destination_port}")
    print(f"Protocol         : {protocol}")


print("\nCapturing 5 packets...\n")

sniff(
    count=5,
    prn=display_packet_features,
    store=False
)

print("\nPacket capture completed.")