from ip_blacklist import block_ip, is_blocked


def prevent_ip(ip_address):
    """
    Safely block an IP using the local blacklist.
    This is currently a prototype/simulation.
    """

    if is_blocked(ip_address):
        return {
            "status": "ALREADY_BLOCKED",
            "ip": ip_address
        }

    block_ip(ip_address)

    return {
        "status": "BLOCKED",
        "ip": ip_address
    }


if __name__ == "__main__":

    test_ip = "192.168.31.200"

    print("Testing prevention system...")
    print()
    print("This is currently a prototype/simulation.")
    print()

    result = prevent_ip(test_ip)

    print("IP     :", result["ip"])
    print("Status :", result["status"])