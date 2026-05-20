class Formatter:
    style_options = { "%", "$", "{"}

    def __init__(self, format="{levelname}:{name}:{message}", style='{', datefmt='%Y-%m-%d %H:%M:%S'):
        if style not in Formatter.style_options:
            raise TypeError(f'invailid formatter style, valid:{Formatter.style_options}')
        
        self.format = format
        self.style = style
        self.datefmt = datefmt