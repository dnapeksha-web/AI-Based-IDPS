BLACKLIST = set()


def block_ip(ip_address):
    """Add an IP address to the blacklist."""

    if ip_address in BLACKLIST:
        return False

    BLACKLIST.add(ip_address)
    return True


def unblock_ip(ip_address):
    """Remove an IP address from the blacklist."""

    if ip_address not in BLACKLIST:
        return False

    BLACKLIST.remove(ip_address)
    return True


def is_blocked(ip_address):
    """Check whether an IP address is blocked."""

    return ip_address in BLACKLIST


def get_blocked_ips():
    """Return all blocked IP addresses."""

    return list(BLACKLIST)


# Test the blacklist
if __name__ == "__main__":

    test_ip = "192.168.31.100"

    print("Blocking IP:", test_ip)

    block_ip(test_ip)

    print("Is blocked:", is_blocked(test_ip))

    print("Blocked IPs:", get_blocked_ips())

    print("Unblocking IP:", test_ip)

    unblock_ip(test_ip)

    print("Is blocked:", is_blocked(test_ip))