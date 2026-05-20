# with open('test.txt') as file:
#     for line in file:
#         print(line, end='')
# # end='' omits the extra newline

with open('test.txt') as file:
    data = file.read()

print(data)