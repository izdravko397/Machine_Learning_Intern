from abc import abstractmethod, ABC
import re

class SaxParser(ABC):
    def parse(self, html: str):
        tag_re = re.compile(r"<(/)?(\w+)([^>]*)>")
        last_match_inx = 0

        for match in tag_re.finditer(html):
            if tag_name := match.group(2):
                if last_match_inx != 0:
                    text = html[last_match_inx : match.start()].strip()
                    if text:
                        self.data(text)

                if match.group(1):
                    self.end_tag(tag_name)
                else:
                    self.start_tag(tag_name, match.group(3).strip())

                last_match_inx = match.end()
       
    @abstractmethod
    def start_tag(self, tag, attrs):
        pass
    
    @abstractmethod
    def end_tag(self, tag):
        pass
    
    @abstractmethod
    def data(self, text):
        pass


class TestParser(SaxParser):
    def start_tag(self, tag, attrs):
        print(f"Encountered a start tag: {tag}")

    def end_tag(self, tag):
        print(f"Encountered an end tag: {tag}")

    def data(self, text):
        print(f"Encountered some data: {text}")


html = '''
<html>
  <head><title class='title'>Test page</title></head>
  <body>
    <h1>Welcome</h1>
    <p>Hello <b>Ivan</b>. How are you?<p>
  </body>
</html>
'''
parser = TestParser()
parser.parse(html)


# start tag: html
# start tag: head
# start tag: title
# text: Test page
# end tag: title
# end tag: head
# start tag: body
# start tag: h1
# text: Welcome
# ---? end tag?
# start tag: p
# text: hello
# start tag: b
# text: Ivan
# end tag: b
# text: . How are you?
# end tag: p
# end tag: h1
# end tag: body
# end tag: html