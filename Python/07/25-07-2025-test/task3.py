def common_letters(str: str, n: int):
    letters_counter = {}

    for letter in str:
        if not letter.isalpha():
            continue

        lower_letter = letter.lower()

        if lower_letter not in letters_counter:
            letters_counter[lower_letter] = 0
        letters_counter[lower_letter] += 1

    res = []
    while letters_counter:
        max_letter = max(letters_counter, key=letters_counter.get)
        res.append(max_letter)
        del letters_counter[max_letter]

        if len(res) == n:
            break

    return res

str1 = 'AAAaaa lla dddZ!'
print(common_letters(str1, 2))