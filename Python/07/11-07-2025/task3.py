def sort_numbers(numbers):
    result = []
    int_numbers = []  # [int(num) for num in numbers.split(', ')]

    for num in numbers.split(', '):
        int_numbers.append(int(num))

    while int_numbers:
        min_num = int_numbers[0]

        for num in int_numbers:
            if num < min_num:
                min_num = num

        result.append(min_num)
        int_numbers.remove(min_num)

    return ', '.join(str(num) for num in result)

print(sort_numbers('1, 3, 1, 2, 5, 2, 1, 3'))

print(sort_numbers('1, 2, 3, 4, 5, 5, 4, 3, 2, 1'))