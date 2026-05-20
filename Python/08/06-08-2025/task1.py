def correct_brackets(str: str):
    close_brackets = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    stack = []

    for bracket in str:
        if val := close_brackets.get(bracket):
            if not stack or stack[-1] != val:
                return False
            
            stack.pop()
        else:
            stack.append(bracket)

    return not stack
        
print(correct_brackets("()"))
print(correct_brackets("()[]{}"))
print(correct_brackets("(]"))
print(correct_brackets("([)]"))
print(correct_brackets("{[]}"))

# "()" => True
# "()[]{}" => True
# "(]" => False
# "([)]" => False
# "{[]}" => True