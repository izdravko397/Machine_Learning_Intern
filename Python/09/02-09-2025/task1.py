import io

class BufferedReader:
    def __init__(self, raw, buffer_size=8192):
        self.raw = raw
        self.buffer_size = buffer_size
        self.offset = 0
        self.content = bytearray()

    def read(self, n=-1):
        if n == -1:
            data = self.raw.read(self.buffer_size)

            while data != b'':
                self.content.extend(data)
                data = self.raw.read(self.buffer_size)
            
            res = self.content
            self.content.clear()
            return res
        
        if self.offset + n > len(self.content):
            self.content = self.content[self.offset:] + self.raw.read(self.buffer_size)
            self.offset = 0

        res = self.content[self.offset : self.offset + n]
        self.offset += n

        return res
    
    def readline(self, n=-1):
        new_line_char = ord('\n')
        line = bytearray()

        while n != 0:
            if len(self.content) == self.offset:
                new_block = self.read(self.buffer_size)
                if not new_block:
                    break

                self.content = new_block
                self.offset = 0

            byte = self.content[self.offset]
            self.offset += 1
            line.append(byte)

            if byte == new_line_char:
                break
            n -= 1
        
        return line

    def close(self):
        self.content.clear()
        self.raw.close()


class TextIOWrapper:
    def __init__(self, buffer: BufferedReader, encoding='utf-8'):
        self.buffer = buffer
        self.encoding = encoding
        self.offset = 0

    def read(self, n=-1):
        file_in_bytes = self.buffer.read(n)
        self.offset += len(file_in_bytes)

        return file_in_bytes.decode(encoding=self.encoding)
    
    def readlines(self):
        res = []

        line = self.buffer.readline()
        while line:
            self.offset += len(line)
            res.append(line.decode(encoding=self.encoding))
            line = self.buffer.readline()

        return res
    
    def readline(self, n=-1):
        line = self.buffer.readline(n)
        self.offset += len(line)
        return line.decode(encoding=self.encoding)
    
    def tell(self):
        return self.offset
    
    def close(self):
        self.buffer.close()

    def __iter__(self):
        return self
    
    def __next__(self):
        line = self.readline()

        if not line:
            raise StopIteration
        return line

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

# raw = io.FileIO('app.log', 'r')
# reader = io.BufferedReader(raw)
# file = io.TextIOWrapper(reader, encoding='utf-8') 

# # with file:
# #   for line in file:
# #     print(line)


raw = io.FileIO('app.log', 'r')
reader = BufferedReader(raw)
file = TextIOWrapper(reader, encoding='utf-8') 

# print(file.readlines())
with file:
  for line in file:
    print(line)
