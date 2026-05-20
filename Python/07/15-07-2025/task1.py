def ip_address_formatting(n):
    MAX_NUM_IP = 4294967295

    if 0 <= n <= MAX_NUM_IP:
        bite32 = f'{n:032b}'

        # b1 = int(bite32[:8], 2)
        # b2 = int(bite32[8:16], 2)
        # b3 = int(bite32[16:24], 2)
        # b4 = int(bite32[24:], 2)    
        # return f'{b1}.{b2}.{b3}.{b4}'

        res = []
        bits = {
            'b1': bite32[:8],
            'b2': bite32[8:16],
            'b3': bite32[16:24],
            'b4': bite32[24:]
        }

        for value in bits.values():
            current_byte = 0
            power = 7
            
            for bit in value:
                current_byte += int(bit) * (2 ** power)
                power -= 1
            res.append(str(current_byte))

        return '.'.join(res)
    
    else:
        raise ValueError



print(ip_address_formatting(65535))
print(ip_address_formatting(256))
print(ip_address_formatting(257))
print(ip_address_formatting(-1))