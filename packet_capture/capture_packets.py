from scapy.all import sniff


def display_packet(packet):
    print(packet.summary())


print("\nCapturing 5 packets...\n")

sniff(
    count=5,
    prn=display_packet,
    store=False
)

print("\nPacket capture completed.")