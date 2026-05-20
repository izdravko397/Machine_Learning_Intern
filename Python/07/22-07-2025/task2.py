# def string_cleaner(str: str):
#     str_list = []

#     for char in str:
#         if char == ' ' and (not str_list or str_list[-1] == ' '):
#             continue

#         str_list.append(char.lower())

#     if str_list[-1] == ' ':
#         str_list.pop()

#     return ''.join(str_list)

# def string_comparison(str1: str, str2: str):
#     clean_str1 = ' '.join(str1.split()).lower()
#     clean_str2 = ' '.join(str2.split()).lower()

#     # clean_str1 = string_cleaner(str1)
#     # clean_str2 = string_cleaner(str2)

#     return clean_str1 == clean_str2

def next_char(str, inx, char_comparison_flag, space_comparison_flag):
    search_next_word = False
    space_inx = ''

    for i in range(inx, len(str)):
        if str[i] != ' ':           
            if search_next_word:
                return str[space_inx], space_inx + 1

            return str[i].lower(), i + 1
        
        if space_comparison_flag:
            continue
        
        if space_inx == '' and char_comparison_flag:
            space_inx = i
            search_next_word = True

    else: 
        return None, 0

def string_comparison(str1: str, str2: str):
    first_char_comparison = False
    last_comparison_space = False
    str1_i = 0
    str2_i = 0

    while True:
        next_char1, str1_i = next_char(str1, str1_i, first_char_comparison, last_comparison_space)
        next_char2, str2_i = next_char(str2, str2_i, first_char_comparison, last_comparison_space)

        if next_char1 != next_char2:
            return False
        else:
            last_comparison_space = False
            if next_char1 == ' ':
                last_comparison_space = True

            if not first_char_comparison:
                first_char_comparison = True
                
            if next_char1 is None and next_char2 is None:
                return True
            

        

            
# print(string_comparison('  abc  ', 'abc'))
# print(string_comparison('ABC', 'abc'))
# print(string_comparison(' ala   bala', 'ala bala'))
# print(string_comparison(' ala bala  ', 'alabala'))
print(string_comparison("a  d", "a   d"))
# print(string_comparison("ala bala", "ala balaaa"))

# "  abc  ", "abc" => true
# "ABC", "abc" => true
# " ala   bala", "ala bala" => true
# " ала bala  ", "alabala" => false