from .loggers import Logger
from .handlers import StreamHandler, FileHandler, Handler
from .formatters import Formatter
from .filter import Filter
import sys

root = Logger()

_first_config = False

def basicConfig(level='WARNING', handler=None, format="%(levelname)s:%(name)s:%(message)s", 
                style='%', datefmt='%Y-%m-%d %H:%M:%S', filename=None, filemode="a", encoding="utf-8"):
    global _first_config
    if _first_config:
        return
    
    _first_config = True

    if handler is None:
        formatter = Formatter(format, style, datefmt)

        if filename is not None:
            handler = FileHandler(fname=filename, mode=filemode, encoding=encoding, formatter=formatter)
        else:
            handler = StreamHandler(formatter=formatter)


    root.addHandler(handler)
    root.setLevel(level)

_loggers = {}

def getLogger(name=''):
    if not name:
        return root
    
    if log := _loggers.get(name):
        return log
    
    logger = Logger(qualname=name)

    while True:
        parent_name = name.rsplit('.', 1)[0]
        if parent_name == name:
            logger.parent = root
            break

        if parent := _loggers.get(parent_name):
            logger.parent = parent
            break

    _loggers[name] = logger
    return logger


def dictConfig(config: dict):
    formatters = {}
    handlers = {}

    if forms := config.get('formatters'):
        for form, params in forms.items():
            format = params.get('format')
            style = params.get('style', '%')
            datefmt = params.get('datefmt', '%Y-%m-%d %H:%M:%S')

            formatters[form] = Formatter(format=format, style=style, datefmt=datefmt)
    
    if hands := config.get('handlers'):
        for hand, params in hands.items():
            h = eval(params.get('class'))
            level = params.get('level', NOTSET)
            formatter = formatters.get('formatter', Formatter())

            if fname := params.get('filename'):
                mode = params.get('mode', 'a')
                encoding = params.get('encoding', 'utf-8')
                handlers[hand] = h(fname=fname, mode=mode, encoding=encoding)
            else:
                stream = eval(params.get('stream', 'sys.stderr'))
                handlers[hand] = h(stream=stream)

            handlers[hand].setLevel(level)
            handlers[hand].setFormatter(formatter)

    if logs := config.get('loggers'):
        for log, params in logs.items():
            logger = getLogger(log)
            logger.setLevel(params.get('level', NOTSET))
            logger.propagate = params.get('propagate', True)

            for hand in params.get('handlers', []):
                logger.addHandler(handlers[hand])
            

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

NOTSET = 0
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50