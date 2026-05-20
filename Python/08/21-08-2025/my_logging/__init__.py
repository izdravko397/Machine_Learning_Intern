from .loggers import Logger
from .handlers import StreamHandler, FileHandler
from .formatters import Formatter
from .filter import Filter

root = Logger()

_first_config = False

def basicConfig(level='WARNING', handler=None, format="{levelname}:{name}:{message}", 
                style='{', datefmt='%Y-%m-%d %H:%M:%S', filename=None, filemode="a", encoding="utf-8"):
    global _first_config
    if _first_config:
        return
    
    _first_config = True

    if handler is None:
        formatter = Formatter(format, style, datefmt)

        if filename is not None:
            handler = FileHandler(fname=filename, mode=filemode, encoding=encoding, formatter=formatter)
        else:
            handler = StreamHandler(formatter)

    root.addHandler(handler)
    root.setLevel(level)

_loggers = {}

def getLogger(name):
    if not name:
        return root
    
    if log := _loggers.get(name):
        return log
    
    logger = Logger(qualname=name)
    _loggers[name] = logger
    return logger

def exception(msg):
    error(msg, exc_info=True)

    
def debug(msg, exc_info=False):
    root.debug(msg, exc_info)

def info(msg, exc_info=False):
    root.info(msg, exc_info)

def warning(msg, exc_info=False):
    root.warning(msg, exc_info)

def error(msg, exc_info=False):
    root.error(msg, exc_info)

def critical(msg, exc_info=False):
    root.critical(msg, exc_info)

NOTSET = 'NOTSET'
DEBUG = 'DEBUG'
INFO = 'INFO'
WARNING = 'WARNING'
ERROR = 'ERROR'
CRITICAL = 'CRITICAL'