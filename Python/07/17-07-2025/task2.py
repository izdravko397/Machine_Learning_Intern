def index_of(s1, s2, start_index=0):
    if start_index >= len(s1) or start_index < 0:
        raise ValueError('Index out of range')
    
    # return s1.find(s2, start_index)

    for inx in range(start_index, len(s1)):
        # if s1[inx: inx + len(s2)] == s2:
        #     return inx
        
        match = True

        if s1[inx] == s2[0] and inx + (len(s2) - 1) <= len(s1) - 1:
            for i in range(1, len(s2)):
                if s1[inx + i] != s2[i]:
                    match = False
                    break
            if match:
                return inx
    else:
        return -1
    
print(index_of("ala bala", "la", 2))
print(index_of("github", "hub"))
print(index_of("microsoft", "google"))
    