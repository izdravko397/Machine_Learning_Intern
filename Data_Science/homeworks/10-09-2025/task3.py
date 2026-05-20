import re

class Node:
    def __init__(self,tag: str, parent=None, children=None, attributes=None, text=None):
        self.parent = parent
        self.tag = tag
        self.children = children or []
        self.attributes = attributes or {}
        self.text = text or ''
        self.t = 0

    def __str__(self):
        a = []
        for ch in self.children:
            ch.t = self.t + 1
            a.append(str(ch))

        childs = '\n'.join(a)
        
        if self.tag != '#text':
            res =  f'{'    ' * self.t}<{self.tag}{f' {' '.join(f'{k} = {v}' for k, v in self.attributes.items())}' if self.attributes else ''}>\n' + childs + f'\n{'    ' * self.t}</{self.tag}>' 
        else:
            res = f'{'    ' * self.t}{self.text}'

        self.t = 0
        return res
    
    def __repr__(self):
        return f'Node(tag = {self.tag}{f', attributes = {self.attributes}' if self.attributes else ''}{f', text = {self.text}' if self.text else ''})'

stack = []
def parse_html(html: str):
    pattern = re.compile(r"<(/)?(\w+)([^>]*)>")
    attr_pattern = re.compile(r'(\w+)\s*=\s*["\']?([^"]+)["\']?')
    matches = pattern.finditer(html)
    current_obj = None
    root = None
    last_match_inx = 0

    for match in matches:
        if tag_name := match.group(2):
            if last_match_inx != 0:
                    text = html[last_match_inx : match.start()].strip()
                    if text:
                        if not stack:
                            raise SyntaxError('No parents')
                        
                        text_obj = Node('#text', stack[-1], text=text)
                        stack[-1].children.append(text_obj)

            if match.group(1):
                if not stack or stack[-1].tag != tag_name:
                    raise SyntaxError
                
                stack.pop()
            else:
                current_obj = Node(tag_name)
                if not stack and root is None:
                    root = current_obj

                if stack:
                    stack[-1].children.append(current_obj)
                    current_obj.parent = stack[-1]
                stack.append(current_obj)

                if not stack and root:
                    raise SyntaxError
                
                attrs = attr_pattern.findall(match.group(3))
                attr_dict = {k: v for k, v in attrs}
                current_obj.attributes = attr_dict

            last_match_inx = match.end()
                
    if stack:
        raise SyntaxError
    
    return root



html = """
<html>
  <body bgcolor="red">
    <h1 id="first-header">My First Heading</h1>
    <p topmargin="12">My first paragraph.</p>
  </body>
</html>
"""

html2 = """
  <body bgcolor="red">
    <h1 id= 'first-header'>My First Heading</h1>
    <p topmargin="12">My first paragraph.</p>
  </body>
"""

# parse_html(html)

root = parse_html(html)
print(repr(root))
# Node(tag=html)
# print(root.children)

rc = root.children
body = rc[0]
print(body)
print(body)
# print(repr(body))
# # Node(tag=body, attributes={bgcolor: red})
# print(repr(body.parent))
# # Node(tag=html)

# bc = body.children
# h1 = bc[0]
# print(repr(h1))
# # Node(tag=h1 attributes={id: "first-header"})

# p = bc[1]
# print(repr(p))
# # Node(tag=p attributes={topmargin="12"})

# pc = p.children
# text_node = pc[0]
# print(repr(text_node))
# # Node(tag=#text text="My first paragraph.")
