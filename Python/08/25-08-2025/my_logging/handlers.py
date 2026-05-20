from abc import ABC, abstractmethod
from .formatters import Formatter
from .filter import Filter
from string import Template
import datetime 
import sys
import traceback
import os

level_val = {
    'NOTSET': 0,
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
}

class Handler(ABC):
    def __init__(self, formatter=Formatter()):
        super().__init__()
        self.level = 0
        self.formatter = formatter
        self.filters = []

    @abstractmethod
    def print_logger(self, record, exc_info=False):
        if self._check_level(record.levelname):
            raise Exception
        
        for f in self.filters:
            if not f.filter(record):
                raise Exception
            
        if 'asctime' in self.formatter.format:
            record.asctime = datetime.datetime.now().strftime(self.formatter.datefmt)

    def setLevel(self, level):
        if isinstance(level, str):
            level = level_val.get(level)
            if level is None:
                raise ValueError('invalid level')

        self.level = level

    def setFormatter(self, formatter):
        if not isinstance(formatter, Formatter):
            raise TypeError('the formatter must be instance from Formatter')
        self.formatter = formatter

    def addFilter(self, filter):
        class Func_filter:
            def __init__(self, filter):
                self.filter = filter

        if not isinstance(filter, Filter):
            filter = Func_filter(filter)

        self.filters.append(filter)

    def _check_level(self, log_level):
        return level_val[log_level] < self.level
    
    @abstractmethod
    def __str__(self):
        pass
    

class StreamHandler(Handler):
    def __init__(self,stream=sys.stderr, formatter=Formatter()):
        super().__init__(formatter)
        self.stream = stream

    def print_logger(self, record, exc_info=False):
        try:
            super().print_logger(record)
        except Exception:
            return
        
        if self.formatter.style == '{':
            self.stream.write(self.formatter.format.format(**record.__dict__) + '\n')
        elif self.formatter.style == '%':
            self.stream.write(self.formatter.format % record.__dict__ + '\n')
        else:
            self.stream.write(Template(self.formatter.format).substitute(**record.__dict__) + '\n')

        if exc_info:
            traceback.print_exception(*sys.exc_info())

    def __str__(self):
        return f'<{self.__class__.__name__} {self.stream.name} ({self.level})>' 


class FileHandler(Handler):
    def __init__(self, fname="app.log", mode="a", encoding="utf-8", formatter=Formatter()):
        super().__init__(formatter)
        self.fname = fname
        self.mode = mode
        self.encoding = encoding
        self.f = open(file=self.fname, mode=self.mode, encoding=self.encoding)

    def print_logger(self, record, exc_info=False):
        try:
            super().print_logger(record)
        except Exception:
            return
        
        if self.formatter.style == '{':
            self.f.write(self.formatter.format.format(**record.__dict__) + '\n')
        elif self.formatter.style == '%':
            self.f.write(self.formatter.format % record.__dict__ + '\n')
        else:
            self.f.write(Template("$levelname").substitute(**record.__dict__) + '\n')
        
        if exc_info:
            self.f.write(''.join(traceback.format_exception(*sys.exc_info())))

        self.f.flush()

    def __del__(self):
        self.f.close()

    def __str__(self):
        return f'<{self.__class__.__name__} {os.path.abspath(self.fname)} ({self.level})>'