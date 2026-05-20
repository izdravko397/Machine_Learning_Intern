from dataclasses import dataclass
import re

def get_node(tag):
    pattern = r"^<(\w+)((\s*\w*\s*=\s*'[\d\w#]*'\s*)+)*>$"
    match = re.search(pattern, tag)

    tag = match.group(1)
    args = match.group(2).split("' ")
    args_dict = {}
    for arg in args:
        key, val = arg.split('=')

        val = val.strip()
        if not val.endswith("'"):
            val += "'"

        args_dict[key.strip()] = val

    return Node(tag, args_dict)


@dataclass
class Node:
    tag: str
    attributes: dict
    
s = "<a href ='#section1' class= 'header' id='12'>"
# s = "<a>"


n = get_node(s)
print(n)
# Node(tag=a, attributes={href: "#section1" class:"header" id:"12")