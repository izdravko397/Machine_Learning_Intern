# да се напише функция split_lines която получава като параметър стринг състоящ 
# се от много редове и връща обект от тип iterable с кото може да вземем отдлените редове на функцията.
# функцията да е оптимална относно изразходвана памет


# def split_lines(str: str):
#     # return (line for line in str.strip().split('\n'))

#     start_inx = 1
#     for i in range(1, len(str)):
#         if str[i] == '\n':
#             yield str[start_inx:i]
#             start_inx = i + 1

# s = '''
# ala
# bala
# portokala
# '''

# lines = split_lines(s)
# for line in lines:
#     print(line)

def split_lines(str: str):
    # return (line for line in str.split('\n'))

    str_len = len(str)
    start_inx = 0
    for i in range(1, str_len):
        if str[i] == '\n' or i + 1 == str_len:
            yield str[start_inx:i]
            start_inx = i + 1
        
s = '''
ala
bala
portokala
'''

lines = split_lines(s)
for line in lines:
    print(line)

# ala
# bala
# portokala