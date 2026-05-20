def student_attendance(str:str):
    TARDINESS_CHAR = 'Z'
    PRESENT_CHAR = 'P'

    absence_counter = 0
    tardiness_counter = 0

    for char in str:
        if char == PRESENT_CHAR:
            tardiness_counter = 0
            continue
        
        elif char == TARDINESS_CHAR:
            tardiness_counter += 1
            if tardiness_counter == 3: 
                return False

        else:
            tardiness_counter = 0
            absence_counter += 1
            if absence_counter == 3: 
                return False

    return True

str1 = 'PPOZZP'
str2 = 'PPOZZZ'
str3 = 'OZOZOZ'

print(student_attendance(str1))
print(student_attendance(str2))
print(student_attendance(str3))

# 'ППОЗЗП' =>  true
# 'PPОЗЗЗ' => false