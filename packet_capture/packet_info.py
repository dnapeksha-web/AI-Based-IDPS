from scapy.all import sniff, IP, IPv6


def display_packet_info(packet):
    print("\n----------------------------")

    # Packet Size
    print(f"Packet Size : {len(packet)} bytes")

    # IPv4
    if packet.haslayer(IP):
        print(f"Source IP      : {packet[IP].src}")
        print(f"Destination IP : {packet[IP].dst}")

    # IPv6
    elif packet.haslayer(IPv6):
        print(f"Source IP      : {packet[IPv6].src}")
        print(f"Destination IP : {packet[IPv6].dst}")

    else:
        print("IP Information : Not Available")


print("\nCapturing 5 packets...\n")

sniff(
    count=5,
    prn=display_packet_info,
    store=False
)

print("\nPacket capture completed.")