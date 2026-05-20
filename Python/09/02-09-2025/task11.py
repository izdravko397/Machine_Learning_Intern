import io

class BufferedReader:
    def __init__(self, raw: io.FileIO, buffer_size=8192):
        self.raw = raw
        self.buffer_size = buffer_size
        self.offset = 0
        self.buffer = bytearray(buffer_size)
        self.valid_bytes = self.raw.readinto(self.buffer)

    def read(self, n=-1):
        if n < 0:
            data = self.buffer[self.offset:self.valid_bytes]

            while self.raw.readinto(self.buffer):
                data.extend(self.buffer)

            data_len = len(data)
            self.offset = data_len
            self.valid_bytes = data_len
            return data
        
        if self.offset + n > self.valid_bytes:
            unread_bytes = self.buffer[self.offset:self.valid_bytes]
            data = unread_bytes + self.raw.read(n - len(unread_bytes))
            self.valid_bytes = self.raw.readinto(self.buffer)
            self.offset = 0
            return data

        data = self.buffer[self.offset:self.offset + n]
        self.offset += n

        valid_bytes_len = self.valid_bytes - self.offset
        if valid_bytes_len > self.buffer_size // 2:
            self.buffer[:valid_bytes_len] = self.buffer[self.offset:self.valid_bytes]
            self.offset = 0

            mv = memoryview(self.buffer)[valid_bytes_len:]
            new_bytes_len = self.raw.readinto(mv)
            self.valid_bytes = valid_bytes_len + new_bytes_len

        return data
    
    def readline(self, n=-1):
        new_line_char = ord('\n')
        line = bytearray()

        while n != 0:
            if self.valid_bytes <= self.offset:
                self.valid_bytes = self.raw.readinto(self.buffer)
                self.offset = 0
                if not self.valid_bytes:
                    break

            byte = self.buffer[self.offset]
            self.offset += 1
            line.append(byte)

            if byte == new_line_char:
                break
            n -= 1

        valid_bytes_len = self.valid_bytes - self.offset
        if valid_bytes_len > self.buffer_size // 2:
            self.buffer[:valid_bytes_len] = self.buffer[self.offset:self.valid_bytes]
            self.offset = 0

            mv = memoryview(self.buffer)[valid_bytes_len:]
            new_bytes_len = self.raw.readinto(mv)
            self.valid_bytes = valid_bytes_len + new_bytes_len
        
        return line

    def close(self):
        self.buffer.clear()
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
    
    def readlines(self, n=float('inf')):
        res = []
        bytes_counter = 0

        while bytes_counter <= n:
            line = self.buffer.readline()
            if not line:
                break

            line_len = len(line)
            self.offset += line_len
            bytes_counter += line_len
            res.append(line.decode(encoding=self.encoding))

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

raw = io.FileIO('app.log', 'r')
reader = io.BufferedReader(raw)
file = io.TextIOWrapper(reader, encoding='utf-8') 
# print(file.readlines(100))
# print(reader.tell())
# print(reader.read())
# print(reader.tell())
# print(reader.read())

# a = bytearray(3)
# print(raw.readinto(a))
# print(a)
# raw.readinto(a)
# print(a)
# print(reader.readline(300000))


# with file:
#   for line in file:
#     print(line)


raw = io.FileIO('app.log', 'r')
reader = BufferedReader(raw, buffer_size=10)
file = TextIOWrapper(reader, encoding='utf-8') 

# print(reader.read(3))
# print(reader.read(3))
# print(reader.read())
# print(reader.offset)
# print(reader.read())


# print(reader.read(2))
# print(reader.read(2000))
# print()
# print(reader.read())
# print(reader.read(4))
# print(file.readlines())
# with file:
#   for line in file:
#     print(line)

print(bool(-1))