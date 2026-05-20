def correct_brackets(str: str):
    close_brackets = {
        ')': '(',
        ']': '[',
        '}': '{'
    }
    open_brackets = {b for b in close_brackets.values()}

    stack = []

    for char in str:
        if char in open_brackets:
            stack.append(char)
        elif val := close_brackets.get(char):
            if not stack or stack.pop() != val:
                return False

    return not stack
        
print(correct_brackets("Cou(n)ter({'blue': 10, 'red': 8})"))
print(correct_brackets("()[]{}"))
print(correct_brackets("(]"))
print(correct_brackets("([Counter({'blue': 10, 'red': 8}))]"))
print(correct_brackets("{[]}"))