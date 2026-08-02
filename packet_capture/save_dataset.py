import csv
import os

from scapy.all import sniff, IP, IPv6, TCP, UDP


# Create the data folder if it does not exist
os.makedirs("data", exist_ok=True)


# CSV file location
CSV_FILE = "data/network_traffic.csv"


# Create the CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "source_ip",
            "destination_ip",
            "source_port",
            "destination_port",
            "protocol",
            "packet_size"
        ])


def save_packet(packet):

    # IP Addresses
    if packet.haslayer(IP):
        source_ip = packet[IP].src
        destination_ip = packet[IP].dst

    elif packet.haslayer(IPv6):
        source_ip = packet[IPv6].src
        destination_ip = packet[IPv6].dst

    else:
        source_ip = "N/A"
        destination_ip = "N/A"

    # Ports and Protocol
    if packet.haslayer(TCP):
        source_port = packet[TCP].sport
        destination_port = packet[TCP].dport
        protocol = "TCP"

    elif packet.haslayer(UDP):
        source_port = packet[UDP].sport
        destination_port = packet[UDP].dport
        protocol = "UDP"

    else:
        source_port = "N/A"
        destination_port = "N/A"
        protocol = "OTHER"

    # Packet Size
    packet_size = len(packet)

    # Save to CSV
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            source_ip,
            destination_ip,
            source_port,
            destination_port,
            protocol,
            packet_size
        ])

    print("Packet saved.")


print("\nCapturing 10 packets...\n")

sniff(
    count=10,
    prn=save_packet,
    store=False
)

print("\nDataset saved successfully.")