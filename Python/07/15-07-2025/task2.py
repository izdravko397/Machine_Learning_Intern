def ip_address_to_num(ip):
    bytes = [int(byte) for byte in ip.split('.')]
    res = 0
    power = 24

    for byte in bytes:
        if 0 <= byte <= 255:
            res += byte * (2 ** power) # byte << power
            power -= 8
        else:
            raise ValueError
        
    return res

print(ip_address_to_num('255.255.255.255'))
print(ip_address_to_num('0.0.0.1'))
print(ip_address_to_num('0.0.1.1'))